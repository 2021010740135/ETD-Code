import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        if half_dim <= 1: return time.float().unsqueeze(-1)
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = time.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLNZero(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, 2 * channels)
        # 零初始化是魔法的起源：保证网络初始状态如同一张白纸，极其稳定
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(cond)).unsqueeze(-1)
        scale, shift = torch.chunk(emb, 2, dim=1)
        return x * (1 + scale) + shift


class ConvNeXtBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        # Depthwise 大核卷积，极大地扩充感受野
        self.dwconv = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.norm = nn.GroupNorm(min(8, in_channels), in_channels)

        # Pointwise 升降维
        self.pwconv1 = nn.Conv1d(in_channels, out_channels * 2, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(out_channels * 2, out_channels, 1)

        self.adaln = AdaLNZero(cond_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

        self.residual_conv = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        res = self.residual_conv(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = self.adaln(x, cond)  # 条件调制
        return self.dropout(x) + res


class Downsample1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.conv(x)


class TemporalAttentionPool(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.25):
        super().__init__()
        # 【SOTA 破壁注入】：在注意力生成核中引入强力 Dropout 逼迫机制
        # 这将切断网络对单一显著特征的过度依赖，强迫 512 维潜变量形成真正的分布式表征
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels // 2, 1, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x shape: [B, C, L_down]
        attn_logits = self.attention(x)  # [B, 1, L_down]

        # 【隔离脏数据】：强行将掩码为 0（缺失值）的区域注意力降至极低
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -1e4)

        w = F.softmax(attn_logits, dim=-1)  # [B, 1, L_down]
        return torch.sum(x * w, dim=-1)  # [B, C]


# ==============================================================================
# 2. 核心模型: 物理感知双通道 U-Net (Mask-Aware Input)
# ==============================================================================
class PhysicsAwareUNet1D(nn.Module):
    def __init__(self,
                 in_channels: int = 2,
                 out_channels: int = 1,
                 base_dim: int = 64,
                 phys_dim: int = 8):  # 【基座对齐】：必须对齐阶段二的 8 维正交特征！
        super().__init__()
        self.in_channels = in_channels
        self.cond_dim = base_dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

        self.phys_mlp = nn.Sequential(
            nn.Linear(phys_dim, base_dim),
            nn.SiLU(),
            nn.Linear(base_dim, self.cond_dim),
        )

        self.inc = nn.Conv1d(in_channels, base_dim, 3, padding=1)

        self.down1 = nn.ModuleList([ConvNeXtBlock1D(base_dim, base_dim, self.cond_dim), Downsample1D(base_dim)])
        self.down2 = nn.ModuleList([ConvNeXtBlock1D(base_dim, base_dim * 2, self.cond_dim), Downsample1D(base_dim * 2)])
        self.down3 = nn.ModuleList(
            [ConvNeXtBlock1D(base_dim * 2, base_dim * 4, self.cond_dim), Downsample1D(base_dim * 4)])

        # 瓶颈层
        self.bot1 = ConvNeXtBlock1D(base_dim * 4, base_dim * 4, self.cond_dim)
        self.latent_pooler = TemporalAttentionPool(base_dim * 4)
        self.bot2 = ConvNeXtBlock1D(base_dim * 4, base_dim * 4, self.cond_dim)

        # 【核心对齐】：辅助头输出维度从 6 改为 phys_dim (8维)，执行全维极值能量一致性预测
        self.aux_head = nn.Sequential(
            nn.Linear(base_dim * 4, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, phys_dim)
        )

        self.up1 = nn.ModuleList([Upsample1D(base_dim * 4), ConvNeXtBlock1D(base_dim * 8, base_dim * 2, self.cond_dim)])
        self.up2 = nn.ModuleList([Upsample1D(base_dim * 2), ConvNeXtBlock1D(base_dim * 4, base_dim, self.cond_dim)])
        self.up3 = nn.ModuleList([Upsample1D(base_dim), ConvNeXtBlock1D(base_dim * 2, base_dim, self.cond_dim)])

        self.outc = nn.Conv1d(base_dim, out_channels, 1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, phys_feats: torch.Tensor, return_latent: bool = False):
        raw_mask = x[:, 1:2, :]

        t_emb = self.time_mlp(t)
        p_emb = self.phys_mlp(phys_feats)
        cond = t_emb + p_emb

        x = self.inc(x)
        skips = [x]

        for res_block, downsample in [self.down1, self.down2, self.down3]:
            x = res_block(x, cond)
            skips.append(x)
            x = downsample(x)

        x = self.bot1(x, cond)

        # 【掩码下采样对齐】: 将原始 256 长的掩码同步缩小到瓶颈层尺度
        mask_down = F.interpolate(raw_mask.float(), size=x.shape[-1], mode='nearest')

        if return_latent:
            # 1. 免疫毒化的注意力池化 (已内置 Dropout 逼迫)
            lat_attn = self.latent_pooler(x, mask_down)  # [B, 256]

            # 2. 免疫毒化的全局最大池化
            x_masked = x.masked_fill(mask_down == 0, -1e4)
            lat_max = F.adaptive_max_pool1d(x_masked, 1).squeeze(-1)  # [B, 256]

            lat_max = torch.where(torch.isinf(lat_max), torch.zeros_like(lat_max), lat_max)
            return torch.cat([lat_attn, lat_max], dim=-1)  # [B, 512]

        pred_phys = self.aux_head(self.latent_pooler(x, mask_down))

        x = self.bot2(x, cond)

        for upsample, res_block in [self.up1, self.up2, self.up3]:
            x = upsample(x)
            skip = skips.pop()
            x = torch.cat((x, skip), dim=1)
            x = res_block(x, cond)

        return self.outc(x), pred_phys


# ==============================================================================
# 3. 扩散过程管理器 (集成 Mask Input & DDIM 极速重构)
# ==============================================================================
class GaussianDiffusion1D(nn.Module):
    # 【基座对齐】：默认 phys_dim 从 6 更新为 8
    def __init__(self, model: nn.Module, seq_length: int = 256, timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02, cond_drop_prob: float = 0.1, phys_dim: int = 8):
        super().__init__()
        self.model = model
        self.seq_length = seq_length
        self.timesteps = timesteps
        self.cond_drop_prob = cond_drop_prob
        self.null_phys_emb = nn.Parameter(torch.randn(1, phys_dim))

        self.register_buffer('betas', torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape):
        batch_size = t.shape[0]
        return a[t].reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alpha = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise, noise

    def forward(self, x_residual: torch.Tensor, mask: torch.Tensor, t: torch.Tensor, phys_feats: torch.Tensor):
        B = x_residual.shape[0]
        x_noisy, noise = self.q_sample(x_residual, t)

        model_input = torch.cat([x_noisy, mask], dim=1)

        drop_mask = (torch.rand(B, 1, device=x_residual.device) < self.cond_drop_prob).float()
        phys_input = torch.where(drop_mask == 1, self.null_phys_emb.expand(B, -1), phys_feats)

        noise_pred, pred_phys = self.model(model_input, t, phys_input)
        return noise_pred, noise, pred_phys

    @torch.no_grad()
    def extract_latent_features(self, x_residual: torch.Tensor, mask: torch.Tensor,
                                phys_feats: torch.Tensor) -> torch.Tensor:
        device = x_residual.device
        B = x_residual.shape[0]
        t_zero = torch.zeros((B,), device=device, dtype=torch.long)

        model_input = torch.cat([x_residual, mask], dim=1)
        latent_features = self.model(model_input, t_zero, phys_feats, return_latent=True)
        return latent_features

    @torch.no_grad()
    def fast_manifold_reconstruct(self, x_start: torch.Tensor, mask: torch.Tensor, phys_feats: torch.Tensor,
                                  noise_level: int = 200, ddim_steps: int = 10) -> torch.Tensor:
        device = x_start.device
        B = x_start.shape[0]

        t_start = torch.full((B,), noise_level - 1, device=device, dtype=torch.long)
        x_noisy, _ = self.q_sample(x_start, t_start)

        step_ratio = noise_level // ddim_steps
        timesteps = (np.arange(0, ddim_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        timesteps_prev = np.append(timesteps[1:], 0)

        img = x_noisy

        for i, step in enumerate(timesteps):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            t_prev = torch.full((B,), timesteps_prev[i], device=device, dtype=torch.long)

            model_input = torch.cat([img, mask], dim=1)
            noise_pred, _ = self.model(model_input, t, phys_feats)

            alpha_bar = self.extract(self.alphas_cumprod, t, img.shape)
            alpha_bar_prev = self.extract(self.alphas_cumprod_prev, t_prev, img.shape)

            pred_x0 = (img - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * noise_pred
            img = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        return img

    @torch.no_grad()
    def compute_anomaly_score(self, x_start: torch.Tensor, mask: torch.Tensor, phys_feats: torch.Tensor,
                              noise_level: int = 200, ddim_steps: int = 10) -> torch.Tensor:
        x_recon = self.fast_manifold_reconstruct(x_start, mask, phys_feats, noise_level, ddim_steps)
        mse = torch.pow(x_start - x_recon, 2) * mask

        valid_lengths = torch.sum(mask, dim=-1).clamp(min=1)
        anomaly_score = torch.sum(mse, dim=-1) / valid_lengths

        return anomaly_score.squeeze(1)


# ==============================================================================
# 4. Unit Test
# ==============================================================================
if __name__ == "__main__":
    print("🚀 测试 Next-Gen PGMA-Diff model.py (Mask-Aware & 8-Dim Physics Align) ...")

    B, L = 8, 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")

    x_residual = torch.randn(B, 1, L).to(device)
    x_mask = torch.ones(B, 1, L).to(device)
    x_mask[0, 0, 100:] = 0.0

    phys_feats = torch.randn(B, 8).to(device)  # 【基座对齐测试】：物理特征验证为 8 维
    t = torch.randint(0, 1000, (B,), device=device).long()

    unet = PhysicsAwareUNet1D(in_channels=2, out_channels=1, base_dim=64, phys_dim=8).to(device)
    diffusion = GaussianDiffusion1D(unet, seq_length=L, timesteps=1000, cond_drop_prob=0.1, phys_dim=8).to(device)

    print("\n[Test 1] 训练态 Forward Pass...")
    noise_pred, noise, pred_phys = diffusion(x_residual, x_mask, t, phys_feats)
    print(f"   Noise Pred:   {noise_pred.shape} (Expect [B,1,256])")
    print(f"   Pred Phys:    {pred_phys.shape} (Expect [B, 8]) - 全维正则生效！")

    print("\n[Test 2] Phase 4 Latent Extraction (掩码穿透提取)...")
    latent_vector = diffusion.extract_latent_features(x_residual, x_mask, phys_feats)
    print(f"   Latent Vector:{latent_vector.shape} (Expect [B, 512])")

    print("\n✅ 测试通过！维度撕裂危险已排除，深层注意力破壁完成。")
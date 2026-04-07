# model.py
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Utility blocks
# ============================================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # Input shape: [B]
        # Output shape: [B, dim]
        device = time.device
        half_dim = self.dim // 2
        if half_dim <= 1:
            return time.float().unsqueeze(-1)
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = time.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FiLM1D(nn.Module):
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, channels * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, W]
        # cond shape: [B, cond_dim]
        scale_shift = self.net(cond).unsqueeze(-1)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        return x * (1.0 + scale) + shift


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.norm = nn.GroupNorm(min(8, in_channels), in_channels)
        self.pw1 = nn.Conv1d(in_channels, out_channels * 2, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv1d(out_channels * 2, out_channels, kernel_size=1)
        self.film = FiLM1D(cond_dim, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.film(x, cond)
        x = self.dropout(x)
        return x + res


class Downsample1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ============================================================
# Conditional U-Net
# ============================================================
class ConditionalUNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError("Input channels must be 3: [noisy_x_log, x_mm, mask].")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_dim = base_dim
        self.cond_dim = base_dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

        self.input_proj = nn.Conv1d(in_channels, base_dim, kernel_size=3, padding=1)

        self.down1_block = ResBlock1D(base_dim, base_dim, self.cond_dim, dropout)
        self.down1 = Downsample1D(base_dim)

        self.down2_block = ResBlock1D(base_dim, base_dim * 2, self.cond_dim, dropout)
        self.down2 = Downsample1D(base_dim * 2)

        self.down3_block = ResBlock1D(base_dim * 2, base_dim * 4, self.cond_dim, dropout)
        self.down3 = Downsample1D(base_dim * 4)

        self.mid1 = ResBlock1D(base_dim * 4, base_dim * 4, self.cond_dim, dropout)
        self.mid2 = ResBlock1D(base_dim * 4, base_dim * 4, self.cond_dim, dropout)

        self.up1 = Upsample1D(base_dim * 4)
        self.up1_block = ResBlock1D(base_dim * 8, base_dim * 2, self.cond_dim, dropout)

        self.up2 = Upsample1D(base_dim * 2)
        self.up2_block = ResBlock1D(base_dim * 4, base_dim, self.cond_dim, dropout)

        self.up3 = Upsample1D(base_dim)
        self.up3_block = ResBlock1D(base_dim * 2, base_dim, self.cond_dim, dropout)

        self.out_proj = nn.Conv1d(base_dim, out_channels, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        return_latent: bool = False,
    ):
        raw_mask = x[:, 2:3, :]
        time_cond = self.time_mlp(t)

        h = self.input_proj(x)
        orig_len = h.shape[-1]
        
        # Sequence length padding for downsampling alignment
        pad_len = (8 - orig_len % 8) % 8
        if pad_len > 0:
            h = F.pad(h, (0, pad_len), mode="replicate")
            raw_mask = F.pad(raw_mask, (0, pad_len), mode="constant", value=0.0)

        h = self.down1_block(h, time_cond)
        s1 = h
        h = self.down1(h)

        h = self.down2_block(h, time_cond)
        s2 = h
        h = self.down2(h)

        h = self.down3_block(h, time_cond)
        s3 = h
        h = self.down3(h)

        h = self.mid1(h, time_cond)
        h = self.mid2(h, time_cond)

        if return_latent:
            mask_down = F.interpolate(raw_mask.float(), size=h.shape[-1], mode="nearest")
            
            # Max Pooling (fill invalid with -1e4)
            h_masked_max = h.masked_fill(mask_down <= 0, -1e4)
            lat_max = F.adaptive_max_pool1d(h_masked_max, 1).squeeze(-1)
            lat_max = torch.where(torch.isinf(lat_max), torch.zeros_like(lat_max), lat_max)
            
            # Mean Pooling (fill invalid with 0.0)
            h_masked_mean = h.masked_fill(mask_down <= 0, 0.0)
            valid_len = mask_down.sum(dim=-1).clamp(min=1.0)
            lat_mean = h_masked_mean.sum(dim=-1) / valid_len
            
            # Output Shape: [B, base_dim * 8]
            return torch.cat([lat_max, lat_mean], dim=-1)

        h = self.up1(h)
        h = torch.cat([h, s3], dim=1)
        h = self.up1_block(h, time_cond)

        h = self.up2(h)
        h = torch.cat([h, s2], dim=1)
        h = self.up2_block(h, time_cond)

        h = self.up3(h)
        h = torch.cat([h, s1], dim=1)
        h = self.up3_block(h, time_cond)

        out = self.out_proj(h)

        if pad_len > 0:
            out = out[:, :, :orig_len]
        return out


# ============================================================
# Diffusion wrapper
# ============================================================
class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        seq_length: int = 256,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.model = model
        self.seq_length = seq_length
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    @staticmethod
    def _ensure_channel(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected tensor with shape [B, W] or [B, 1, W], got {tuple(x.shape)}")
        return x

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
        batch_size = t.shape[0]
        return a[t].reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise, noise

    def _build_model_input(self, x_noisy: torch.Tensor, x_mm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.cat([x_noisy, x_mm, mask], dim=1)

    def forward(
        self,
        x_log: torch.Tensor,
        x_mm: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x_log = self._ensure_channel(x_log)
        x_mm = self._ensure_channel(x_mm)
        mask = self._ensure_channel(mask).float()

        x_noisy, noise = self.q_sample(x_log, t)
        model_input = self._build_model_input(x_noisy, x_mm, mask)
        noise_pred = self.model(model_input, t)

        obs_weight = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        diff_loss = ((noise_pred - noise) ** 2 * mask).sum(dim=-1, keepdim=True) / obs_weight
        diff_loss = diff_loss.mean()

        return {
            "noise_pred": noise_pred,
            "noise": noise,
            "loss": diff_loss,
        }

    @torch.no_grad()
    def extract_latent_features(
        self,
        x_log: torch.Tensor,
        x_mm: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x_log = self._ensure_channel(x_log)
        x_mm = self._ensure_channel(x_mm)
        mask = self._ensure_channel(mask).float()
        B = x_log.shape[0]
        t_zero = torch.zeros((B,), device=x_log.device, dtype=torch.long)
        model_input = self._build_model_input(x_log, x_mm, mask)
        return self.model(model_input, t_zero, return_latent=True)

    @torch.no_grad()
    def fast_manifold_reconstruct(
        self,
        x_log: torch.Tensor,
        x_mm: torch.Tensor,
        mask: torch.Tensor,
        noise_level: int = 150,
        ddim_steps: int = 10,
    ) -> torch.Tensor:
        x_log = self._ensure_channel(x_log)
        x_mm = self._ensure_channel(x_mm)
        mask = self._ensure_channel(mask).float()

        B = x_log.shape[0]
        device = x_log.device
        t_start = torch.full((B,), noise_level - 1, device=device, dtype=torch.long)
        img, _ = self.q_sample(x_log, t_start)

        step_ratio = max(noise_level // ddim_steps, 1)
        timesteps = (np.arange(0, ddim_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        timesteps_prev = np.append(timesteps[1:], 0)

        for i, step in enumerate(timesteps):
            t = torch.full((B,), int(step), device=device, dtype=torch.long)
            t_prev = torch.full((B,), int(timesteps_prev[i]), device=device, dtype=torch.long)

            model_input = self._build_model_input(img, x_mm, mask)
            noise_pred = self.model(model_input, t)

            alpha_bar = self.extract(self.alphas_cumprod, t, img.shape)
            alpha_bar_prev = self.extract(self.alphas_cumprod_prev, t_prev, img.shape)

            pred_x0 = (img - torch.sqrt(1.0 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * noise_pred
            img = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        return img

    @torch.no_grad()
    def compute_anomaly_score(
        self,
        x_log: torch.Tensor,
        x_mm: torch.Tensor,
        mask: torch.Tensor,
        noise_level: int = 150,
        ddim_steps: int = 10,
        top_k_ratio: float = 0.15 
    ) -> torch.Tensor:
        x_log = self._ensure_channel(x_log)
        mask = self._ensure_channel(mask).float()
        
        x_recon = self.fast_manifold_reconstruct(x_log, x_mm, mask, noise_level, ddim_steps)
        mse = ((x_log - x_recon) ** 2) * mask
        
        B, C, W = mse.shape
        mse_flat = mse.view(B, W)
        mask_flat = mask.view(B, W)
        
        score_list = []
        for i in range(B):
            valid_mse = mse_flat[i][mask_flat[i] > 0]
            if len(valid_mse) == 0:
                score_list.append(0.0)
                continue
            
            k = max(int(len(valid_mse) * top_k_ratio), 1)
            topk_mse, _ = torch.topk(valid_mse, k)
            score_list.append(topk_mse.mean().item())
            
        return torch.tensor(score_list, device=x_log.device, dtype=torch.float32)

def unpack_diffusion_batch(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"]
    if x.ndim != 3 or x.shape[1] < 3:
        raise ValueError(f"Expected batch['x'] with shape [B, 3, W], got {tuple(x.shape)}")
    x_log = x[:, 0:1, :]
    x_mm = x[:, 1:2, :]
    mask = x[:, 2:3, :]
    return x_log, x_mm, mask

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, W = 8, 256

    x_log = torch.randn(B, 1, W, device=device)
    x_mm = torch.sigmoid(torch.randn(B, 1, W, device=device))
    mask = torch.ones(B, 1, W, device=device)
    mask[0, 0, 128:] = 0.0
    t = torch.randint(0, 1000, (B,), device=device)

    model = ConditionalUNet1D(in_channels=3, out_channels=1, base_dim=32).to(device)
    diffusion = GaussianDiffusion1D(model, seq_length=W, timesteps=1000).to(device)

    out = diffusion(x_log, x_mm, mask, t)
    latent = diffusion.extract_latent_features(x_log, x_mm, mask)
    score = diffusion.compute_anomaly_score(x_log, x_mm, mask, noise_level=100, ddim_steps=5)
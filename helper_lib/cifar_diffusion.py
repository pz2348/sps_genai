import math

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(timesteps, dimension=32, max_period=10_000):
    """Embed one diffusion timestep per image."""
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device) / half
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.cos(), angles.sin()), dim=1)
    return F.pad(embedding, (0, dimension % 2))


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(32, out_channels)
        self.output_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm = nn.GroupNorm(4, out_channels)

    def forward(self, image, time):
        features = self.input_conv(image)
        features = features + self.time_projection(time)[:, :, None, None]
        return self.output_conv(F.silu(self.norm(features)))


class CIFARDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 32))
        self.down1 = TimeBlock(3, 16)
        self.down2 = TimeBlock(16, 24)
        self.middle = TimeBlock(24, 32)
        self.up2 = TimeBlock(32 + 24, 24)
        self.up1 = TimeBlock(24 + 16, 16)
        self.output = nn.Conv2d(16, 3, 1)

    def forward(self, images, timesteps):
        time = self.time_mlp(timestep_embedding(timesteps))
        skip1 = self.down1(images, time)
        skip2 = self.down2(F.avg_pool2d(skip1, 2), time)
        features = self.middle(F.avg_pool2d(skip2, 2), time)
        features = self.up2(
            torch.cat((F.interpolate(features, scale_factor=2), skip2), dim=1), time
        )
        features = self.up1(
            torch.cat((F.interpolate(features, scale_factor=2), skip1), dim=1), time
        )
        return self.output(F.silu(features))


class CIFARDiffusion(nn.Module):
    def __init__(self, steps=100):
        super().__init__()
        self.denoiser = CIFARDenoiser()
        betas = torch.linspace(0.0001, 0.10, steps)
        alphas = 1.0 - betas
        alpha_bars = alphas.cumprod(dim=0)
        previous = F.pad(alpha_bars[:-1], (1, 0), value=1)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", betas * (1 - previous) / (1 - alpha_bars))

    def training_loss(self, clean_images):
        timesteps = torch.randint(
            len(self.betas), (len(clean_images),), device=clean_images.device
        )
        noise = torch.randn_like(clean_images)
        alpha_bar = self.alpha_bars[timesteps, None, None, None]
        noisy_images = alpha_bar.sqrt() * clean_images + (1 - alpha_bar).sqrt() * noise
        predicted_noise = self.denoiser(noisy_images, timesteps)
        return F.mse_loss(predicted_noise, noise)

    @torch.inference_mode()
    def sample(self, count, generator=None):
        device = self.betas.device
        images = torch.randn((count, 3, 32, 32), device=device, generator=generator)
        for step in reversed(range(len(self.betas))):
            timesteps = torch.full((count,), step, device=device, dtype=torch.long)
            noise = self.denoiser(images, timesteps)
            mean = (
                images - self.betas[step] * noise / (1 - self.alpha_bars[step]).sqrt()
            ) / self.alphas[step].sqrt()
            images = mean
            if step > 0:
                noise = torch.randn(images.shape, device=device, generator=generator)
                images = images + self.posterior_variance[step].sqrt() * noise
        return images.clamp(-1, 1)

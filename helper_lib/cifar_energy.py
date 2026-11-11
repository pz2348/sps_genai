import torch
from torch import nn


class CIFAREnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, images):
        return self.network(images).squeeze(1)

    def sample(self, initial_images, steps, generator=None):
        images = initial_images.detach()
        for _ in range(steps):
            with torch.enable_grad():
                images = images.detach().requires_grad_(True)
                # Only pixel gradients are returned. Parameter .grad fields are untouched.
                gradient, = torch.autograd.grad(self(images).sum(), images)
            with torch.no_grad():
                noise = torch.randn(images.shape, device=images.device, generator=generator)
                images = (images - 8.0 * gradient.clamp(-0.03, 0.03) + 0.025 * noise)
                images = images.clamp(-1, 1)
        return images.detach()

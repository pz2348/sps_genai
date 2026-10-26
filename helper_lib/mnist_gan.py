
import torch
import torch.nn as nn


# ==========================================
# Assignment 3 - MNIST Generator
# ==========================================

class MNISTGenerator(nn.Module):
    def __init__(self, noise_dim=100):
        super(MNISTGenerator, self).__init__()

        self.noise_dim = noise_dim

        # Fully connected layer: 100 -> 7*7*128
        self.fc = nn.Linear(noise_dim, 7 * 7 * 128)

        self.network = nn.Sequential(
            # 128 x 7 x 7 -> 64 x 14 x 14
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 64 x 14 x 14 -> 1 x 28 x 28
            nn.ConvTranspose2d(
                64, 1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        return self.network(x)


# ==========================================
# Assignment 3 - MNIST Discriminator
# ==========================================

class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super(MNISTDiscriminator, self).__init__()

        self.network = nn.Sequential(
            # 1 x 28 x 28 -> 64 x 14 x 14
            nn.Conv2d(
                1, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2),

            # 64 x 14 x 14 -> 128 x 7 x 7
            nn.Conv2d(
                64, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# Assignment 3 - GAN Wrapper
# ==========================================

class MNISTGAN(nn.Module):
    def __init__(self, noise_dim=100):
        super(MNISTGAN, self).__init__()

        self.noise_dim = noise_dim
        self.generator = MNISTGenerator(noise_dim)
        self.discriminator = MNISTDiscriminator()


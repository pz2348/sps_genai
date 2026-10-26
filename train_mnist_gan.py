
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from helper_lib.mnist_gan import MNISTGAN


# ==========================================
# Configuration
# ==========================================

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.0002
NOISE_DIM = 100

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# Load MNIST Dataset
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Training images:", len(train_dataset))


# ==========================================
# Create GAN Model
# ==========================================

gan = MNISTGAN(noise_dim=NOISE_DIM).to(device)

generator = gan.generator
discriminator = gan.discriminator

criterion = nn.BCELoss()

optimizer_g = optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)

optimizer_d = optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999)
)


# ==========================================
# Train GAN
# ==========================================

for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()

    total_g_loss = 0.0
    total_d_loss = 0.0

    for real_images, _ in train_loader:
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # ----------------------------------
        # Train Discriminator
        # ----------------------------------

        noise = torch.randn(
            batch_size, NOISE_DIM, device=device
        )

        fake_images = generator(noise)

        optimizer_d.zero_grad()

        real_outputs = discriminator(real_images)
        fake_outputs = discriminator(fake_images.detach())

        loss_real = criterion(real_outputs, real_labels)
        loss_fake = criterion(fake_outputs, fake_labels)

        loss_d = loss_real + loss_fake

        loss_d.backward()
        optimizer_d.step()

        # ----------------------------------
        # Train Generator
        # ----------------------------------

        noise = torch.randn(
            batch_size, NOISE_DIM, device=device
        )

        optimizer_g.zero_grad()

        generated_images = generator(noise)
        outputs = discriminator(generated_images)

        loss_g = criterion(outputs, real_labels)

        loss_g.backward()
        optimizer_g.step()

        total_d_loss += loss_d.item()
        total_g_loss += loss_g.item()

    avg_d_loss = total_d_loss / len(train_loader)
    avg_g_loss = total_g_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}, "
        f"D Loss: {avg_d_loss:.4f}, "
        f"G Loss: {avg_g_loss:.4f}"
    )


# ==========================================
# Save Trained Generator
# ==========================================

torch.save(
    generator.state_dict(),
    "mnist_generator.pth"
)

print("Generator saved to mnist_generator.pth")


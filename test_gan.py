
import torch
from torch.utils.data import DataLoader, TensorDataset

from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples


# ==========================================
# Configuration
# ==========================================

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# Create Small Test Dataset
# ==========================================

# 8 random RGB images, normalized to [-1, 1]
images = torch.rand(8, 3, 64, 64) * 2 - 1

dataset = TensorDataset(images)

data_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


# ==========================================
# Create GAN Model
# ==========================================

model = get_model("GAN")


# ==========================================
# Test GAN Training
# ==========================================

model = train_gan(
    model=model,
    data_loader=data_loader,
    device=device,
    epochs=1,
    n_critic=1
)


# ==========================================
# Generate Images
# ==========================================

generate_samples(
    model=model,
    device=device,
    num_samples=5
)

print("GAN test completed successfully!")

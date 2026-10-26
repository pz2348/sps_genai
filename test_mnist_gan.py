
import torch
import matplotlib.pyplot as plt

from helper_lib.mnist_gan import MNISTGenerator


# ==========================================
# Configuration
# ==========================================

NOISE_DIM = 100
NUM_IMAGES = 10

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# ==========================================
# Load Trained Generator
# ==========================================

generator = MNISTGenerator(noise_dim=NOISE_DIM).to(device)

generator.load_state_dict(
    torch.load(
        "mnist_generator.pth",
        map_location=device,
        weights_only=True
    )
)

generator.eval()

print("Trained Generator loaded successfully!")


# ==========================================
# Generate Handwritten Digits
# ==========================================

with torch.no_grad():
    noise = torch.randn(NUM_IMAGES, NOISE_DIM, device=device)
    generated_images = generator(noise)

# Convert from [-1, 1] to [0, 1]
generated_images = (generated_images + 1) / 2
generated_images = generated_images.cpu()


# ==========================================
# Display Images
# ==========================================

fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    ax.imshow(
        generated_images[i].squeeze(),
        cmap="gray",
        vmin=0,
        vmax=1
    )
    ax.axis("off")

plt.suptitle("MNIST GAN - Generated Handwritten Digits")
plt.tight_layout()
plt.show()

print("Image generation test completed!")

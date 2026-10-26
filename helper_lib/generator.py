
import torch
import matplotlib.pyplot as plt
import math


# ==========================================
# Generate Samples - VAE and GAN
# ==========================================

def generate_samples(model, device='cpu', num_samples=10):
    model.to(device)
    model.eval()

    with torch.no_grad():

        # GAN: Generate images using the Generator
        if hasattr(model, "generator"):
            z = torch.randn(
                num_samples,
                model.z_dim,
                1,
                1,
                device=device
            )

            generated_images = model.generator(z)

        # VAE: Generate images using the Decoder
        elif hasattr(model, "decoder"):
            z = torch.randn(
                num_samples,
                2,
                device=device
            )

            generated_images = model.decoder(z)

        else:
            raise ValueError(
                "Model must contain a generator or decoder."
            )

        # Move images to CPU
        generated_images = generated_images.cpu()

    # Calculate grid size
    cols = 5
    rows = math.ceil(num_samples / cols)

    # Create figure
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(10, 2 * rows),
        squeeze=False
    )

    axes = axes.flatten()

    # Display generated images
    for i in range(num_samples):
        image = generated_images[i]

        if image.shape[0] == 3:
            # GAN RGB image: [-1, 1] -> [0, 1]
            image = (image + 1) / 2
            image = image.permute(1, 2, 0)
            axes[i].imshow(image.clamp(0, 1))

        else:
            # VAE grayscale image
            axes[i].imshow(
                image.squeeze(),
                cmap='gray',
                vmin=0,
                vmax=1
            )

        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

import torch
import matplotlib.pyplot as plt
import math


def generate_samples(model, device='cpu', num_samples=10):
    model.to(device)
    model.eval()

    with torch.no_grad():

        # Generate random points in the latent space
        z = torch.randn(
            num_samples,
            2
        ).to(device)

        # Generate images using the decoder
        generated_images = model.decoder(z)

        # Move images back to CPU
        generated_images = generated_images.cpu()

    # Calculate grid size
    cols = 5
    rows = math.ceil(num_samples / cols)

    # Create the figure
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(10, 2 * rows)
    )

    axes = axes.flatten()

    # Display generated images
    for i in range(num_samples):
        axes[i].imshow(
            generated_images[i].squeeze(),
            cmap='gray'
        )
        axes[i].axis('off')

    # Hide unused spaces
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
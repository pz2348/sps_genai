import torch
from torchvision import datasets, transforms


def get_data_loader(
    data_dir,
    batch_size=32,
    train=True,
    vae=False
):
    # ==========================================
    # Transform
    # ==========================================

    if vae:
        # VAE:
        # Pad 28x28 images to 32x32
        # Keep pixel values between 0 and 1
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad(2),
            transforms.ToTensor()
        ])

    else:
        # FCNN / CNN / EnhancedCNN
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,),
                (0.5,)
            )
        ])

    # ==========================================
    # Dataset
    # ==========================================

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    # ==========================================
    # Data Loader
    # ==========================================

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return loader
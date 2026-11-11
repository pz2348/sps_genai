import argparse
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.cifar_diffusion import CIFARDiffusion
from helper_lib.cifar_energy import CIFAREnergy


def train_energy(model, images, replay):
    count = len(images)
    indices = torch.randint(len(replay), (count,), device=images.device)
    initial_images = torch.rand_like(images).mul(2).sub(1)
    refresh = torch.rand((count, 1, 1, 1), device=images.device) < 0.05
    initial_images = torch.where(refresh, initial_images, replay[indices])
    negative = model.sample(initial_images, steps=8)
    replay[indices] = negative

    real_energy = model(images)
    negative_energy = model(negative)
    contrast = (real_energy - negative_energy).mean()
    penalty = (real_energy.square() + negative_energy.square()).mean()
    return contrast + 0.1 * penalty


def save_weights(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(model.state_dict(), temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Train one CIFAR-10 image generator.")
    parser.add_argument("model", choices=("diffusion", "energy"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or (
        args.max_batches is not None and args.max_batches < 1
    ):
        parser.error("epochs, batch size, threads and max batches must be positive")

    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.CIFAR10(
        root=args.data,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    if args.model == "diffusion":
        model = CIFARDiffusion().to(device)
        replay = None
    else:
        model = CIFAREnergy().to(device)
        replay = torch.rand((2048, 3, 32, 32), device=device).mul_(2).sub_(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    total_batches = min(args.max_batches or len(loader), len(loader))
    print(f"Training {args.model} on {device} for {total_batches} batches per epoch", flush=True)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for step, (images, _) in enumerate(loader, start=1):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            if replay is None:
                loss = model.training_loss(images)
            else:
                loss = train_energy(model, images, replay)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(f"{args.model} training diverged at epoch {epoch}, batch {step}")
            loss.backward()
            optimizer.step()
            total_loss += loss_value
            if step % 50 == 0 or step == total_batches:
                print(f"Epoch {epoch}, batch {step}/{total_batches}, loss {total_loss / step:.4f}",
                      flush=True)
            if step == total_batches:
                break

        save_weights(model, args.output / f"cifar_{args.model}.pth")


if __name__ == "__main__":
    main()

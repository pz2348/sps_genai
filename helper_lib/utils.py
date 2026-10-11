import torch


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


def load_model(model, file_path, device='cpu'):
    model.load_state_dict(
        torch.load(file_path, map_location=device)
    )
    model.to(device)
    return model
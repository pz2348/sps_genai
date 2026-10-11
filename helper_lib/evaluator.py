import torch


def evaluate_model(model, data_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy
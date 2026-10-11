import torch


# ==========================================
# Regular Model Training
# ==========================================

def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(data_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {average_loss:.4f}"
        )

    return model


# ==========================================
# VAE Training
# ==========================================

def train_vae_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device='cpu',
    epochs=10
):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for inputs, labels in data_loader:
            inputs = inputs.to(device)

            optimizer.zero_grad()

            recon, mu, logvar = model(inputs)

            loss = criterion(
                recon,
                inputs,
                mu,
                logvar
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(data_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"VAE Loss: {average_loss:.4f}"
        )

    print("Finished Training")

    return model

# ==========================================
# VAE Loss Function
# ==========================================

def vae_loss_function(recon_x, x, mu, logvar):
    beta = 500

    BCE = torch.nn.functional.binary_cross_entropy(
        recon_x,
        x,
        reduction='sum'
    )

    KLD = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return beta * BCE + KLD
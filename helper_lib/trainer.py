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



# ==========================================
# Module 6 - GAN Training (WGAN)
# ==========================================

def train_gan(
    model,
    data_loader,
    criterion=None,
    optimizer=None,
    device='cpu',
    epochs=10,
    n_critic=5,
    clip_value=0.01,
    lr=5e-5
):
    model = model.to(device)
    model.train()

    generator = model.generator
    critic = model.critic

    opt_gen = torch.optim.RMSprop(
        generator.parameters(), lr=lr
    )
    opt_critic = torch.optim.RMSprop(
        critic.parameters(), lr=lr
    )

    for epoch in range(epochs):
        total_gen_loss = 0.0
        total_critic_loss = 0.0
        num_batches = 0

        for batch in data_loader:
            # Support datasets with or without labels
            real = batch[0] if isinstance(batch, (tuple, list)) else batch
            real = real.to(device)
            batch_size = real.size(0)

            # Train Critic multiple times
            for _ in range(n_critic):
                noise = torch.randn(
                    batch_size, model.z_dim, 1, 1,
                    device=device
                )

                fake = generator(noise).detach()

                critic_real = critic(real).mean()
                critic_fake = critic(fake).mean()

                loss_critic = critic_fake - critic_real

                opt_critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                # Weight clipping for WGAN
                with torch.no_grad():
                    for param in critic.parameters():
                        param.clamp_(-clip_value, clip_value)

            # Train Generator
            noise = torch.randn(
                batch_size, model.z_dim, 1, 1,
                device=device
            )

            fake = generator(noise)
            loss_gen = -critic(fake).mean()

            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            total_critic_loss += loss_critic.item()
            total_gen_loss += loss_gen.item()
            num_batches += 1

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Critic Loss: {total_critic_loss / num_batches:.4f}, "
            f"Generator Loss: {total_gen_loss / num_batches:.4f}"
        )

    print("Finished GAN Training")
    return model

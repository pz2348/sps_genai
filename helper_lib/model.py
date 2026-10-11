import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# FCNN
# ==========================================

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


# ==========================================
# CNN
# ==========================================

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ==========================================
# Enhanced CNN
# ==========================================

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Dropout(0.25)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ==========================================
# Variational Encoder
# ==========================================

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(VariationalEncoder, self).__init__()

        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1
        )

        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=2, padding=1
        )

        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=1
        )

        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(
            128 * 4 * 4, latent_dim
        )

        self.fc_logvar = nn.Linear(
            128 * 4 * 4, latent_dim
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.flatten(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


# ==========================================
# Decoder
# ==========================================

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(
            2, 128 * 4 * 4
        )

        self.convtrans1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        self.convtrans2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        self.convtrans3 = nn.ConvTranspose2d(
            32,
            1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

    def forward(self, x):
        x = self.fc(x)

        x = x.view(
            -1, 128, 4, 4
        )

        x = F.relu(
            self.convtrans1(x)
        )

        x = F.relu(
            self.convtrans2(x)
        )

        x = torch.sigmoid(
            self.convtrans3(x)
        )

        return x


# ==========================================
# VAE
# ==========================================

class VAE(nn.Module):
    def __init__(self, encoder, decoder, beta=1.0):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(
            0.5 * logvar
        )

        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)

        z = self.reparameterize(
            mu, logvar
        )

        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar



# ==========================================
# Assignment 2 - CIFAR10 CNN
# ==========================================

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            # Input: 3 x 64 x 64
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Input: 16 x 32 x 32
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
# ==========================================
# Get Model
# ==========================================

def get_model(model_name):
    if model_name == "FCNN":
        return FCNN()

    elif model_name == "CNN":
        return CNN()

    elif model_name == "EnhancedCNN":
        return EnhancedCNN()

    elif model_name == "VAE":
        encoder = VariationalEncoder()
        decoder = Decoder()

        return VAE(
            encoder,
            decoder
        )
        
    elif model_name == "CIFAR10CNN":
        return CIFAR10CNN()
    
    else:
        raise ValueError(
            "Model name must be FCNN, CNN, EnhancedCNN, VAE, or CIFAR10CNN"
        )
        
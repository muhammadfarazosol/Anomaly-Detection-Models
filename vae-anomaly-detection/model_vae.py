# model_vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Defines the Variational Autoencoder (VAE) architecture.
    """
    def __init__(self, n_features, encoding_dim=32):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(n_features, 128)
        self.fc21 = nn.Linear(128, encoding_dim)  # For mean
        self.fc22 = nn.Linear(128, encoding_dim)  # For log variance
        # Decoder
        self.fc3 = nn.Linear(encoding_dim, 128)
        self.fc4 = nn.Linear(128, n_features)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        """
        The reparameterization trick to allow backpropagation through a random process.
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3)) # Using sigmoid as output is scaled 0-1

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    """
    Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence.
    """
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
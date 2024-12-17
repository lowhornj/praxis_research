import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(32, latent_dim)  # Mean of latent space
        self.log_var_layer = nn.Linear(32, latent_dim)  # Log-variance of latent space
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        log_var = self.log_var_layer(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var

def train_vae(model, data, epochs=50, batch_size=32, learning_rate=0.001):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss(reduction='sum')  # Reconstruction loss
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batch = torch.tensor(batch, dtype=torch.float32)
            
            optimizer.zero_grad()
            x_reconstructed, mu, log_var = model(batch)
            
            # Loss = reconstruction loss + KL divergence
            reconstruction_loss = loss_fn(x_reconstructed, batch)
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = reconstruction_loss + kl_divergence
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        if epoch%10==0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

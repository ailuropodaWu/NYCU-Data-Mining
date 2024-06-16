import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.mean_layer = nn.Linear(latent_dim, 1)
        self.logvar_layer = nn.Linear(latent_dim, 1)
        
        self.decoder = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to('cuda' if torch.cuda.is_available() else 'cpu')      
        z = mean + var * epsilon
        return z
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

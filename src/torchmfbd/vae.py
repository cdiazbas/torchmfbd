import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# `from .vae import *` in __init__ must not rebind torchmfbd.optim (scipy.optimize) to torch.optim
__all__ = ['LinearVAE', 'ConvVAE2D', 'vae_loss_function']

class LinearVAE(nn.Module):
    def __init__(self, in_features: int, latent_dim: int, nested_dims: list = None):
        super(LinearVAE, self).__init__()

        self.nested_dims = nested_dims
        
        # Linear Encoder projections for mean and log-variance
        self.fc_mu = nn.Linear(in_features, latent_dim, bias=True)
        self.fc_logvar = nn.Linear(in_features, latent_dim, bias=True)
        
        # Linear Decoder projection mapping back to original dimension
        self.fc_decoder = nn.Linear(latent_dim, in_features, bias=False)

        self.act = nn.Softplus()  # Using Softplus to ensure positive outputs for variance

    def encode(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick: z = mu + sigma * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.fc_decoder(z)

    def forward(self, x):        
        mu, logvar = self.encode(x)        
        z = self.reparameterize(mu, logvar)

        # Matryoshka-style nested latent dimensions: decode only the first k dimensions of z for each k in nested_dims
        if self.nested_dims is None:
            x_recon = self.decode(z)
        else:
            x_recon = {}
            for k in self.nested_dims:
                # Mask out dimensions beyond index k
                mask = torch.zeros_like(z)
                mask[:, :k] = 1.0
                z_masked = z * mask
                
                # Decode masked latent vector
                x_recon[k] = self.decode(z_masked)
                        
        return x_recon, mu, logvar, z


class ConvVAE2D(nn.Module):
    def __init__(self, in_channels: int = 1, img_size: int = 64, latent_dim: int = 32, nested_dims: list = None):
        super(ConvVAE2D, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.nested_dims = nested_dims
        
        # ----------------------------------------------------
        # 1. Non-Linear 2D Convolutional Encoder
        # ----------------------------------------------------
        # Input shape: [Batch, in_channels, img_size, img_size]
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # -> [B, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),          # -> [B, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),         # -> [B, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),        # -> [B, 256, 4, 4]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # Spatial dimensions after 4 downsamplings (stride 2): img_size / (2^4)
        self.spatial_dim = img_size // 16
        self.feature_dim = 256 * self.spatial_dim * self.spatial_dim

        # Linear projections for latent parameters
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

        # ----------------------------------------------------
        # 2. Non-Linear 2D Transposed Convolutional Decoder
        # ----------------------------------------------------
        # Project latent vector z back to flattened feature shape
        self.decoder_fc = nn.Linear(latent_dim, self.feature_dim)

        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> [B, 128, 8, 8]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> [B, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> [B, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1) # -> [B, in_channels, 64, 64]
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        # Reshape flat linear tensor back to 2D feature map [B, 256, H_small, W_small]
        h = h.view(h.size(0), 256, self.spatial_dim, self.spatial_dim)
        logits = self.decoder_deconv(h)
        
        # Raw logits: the standardised log-PSF. Deconvolution.compute_psfs_vae de-standardises
        # them and takes a softmax over the pixels, which makes the PSF positive with unit sum
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # Matryoshka-style nested latent dimensions: decode only the first k dimensions of z for each k in nested_dims
        if self.nested_dims is None:
            x_recon = self.decode(z)
        else:
            x_recon = {}
            for k in self.nested_dims:
                # Mask out dimensions beyond index k
                mask = torch.zeros_like(z)
                mask[:, :k] = 1.0
                z_masked = z * mask
                
                # Decode masked latent vector
                x_recon[k] = self.decode(z_masked)
        
        return x_recon, mu, logvar, z
    
def vae_loss_function(x_recon, x, mu, logvar, beta=1.0):
    """
    Computes ELBO loss: Reconstruction Loss (MSE) + beta * KL Divergence.
    """
    # Sum of squared errors over features

    # If x_recon is a dictionary (nested latent dimensions), compute reconstruction loss for each and sum    
    if isinstance(x_recon, dict):
        recon_loss = 0.0
        for k, xi in x_recon.items():
            recon_loss += nn.functional.mse_loss(xi, x, reduction='sum') / len(x_recon)
    else:
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
    
    # Closed-form KL divergence for Gaussian variational posterior vs standard normal
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Average total loss over the batch size
    return recon_loss / x.size(0), beta * kl_loss / x.size(0), (recon_loss + beta * kl_loss) / x.size(0)

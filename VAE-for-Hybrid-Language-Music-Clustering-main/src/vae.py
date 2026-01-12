import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class TextDataset(Dataset):
    def __init__(self, features):
        self.features = torch.from_numpy(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

def train_vae(features, hidden_dim=128, latent_dim=10, epochs=100, batch_size=16, lr=0.001):
    dataset = TextDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = VAE(features.shape[1], hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            recon_loss = nn.MSELoss()(recon, batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss / batch.size(0)
            loss.backward()
            optimizer.step()
    return model

class ConvVAE(nn.Module):
    """
    Convolutional VAE for text (character-level or token-level)
    Input: (batch, seq_len, embedding_dim) or flattened
    """
    def __init__(self, vocab_size=500, embed_dim=64, seq_len=512, latent_dim=32):
        super(ConvVAE, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder: Conv1D layers
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size after convs (depends on seq_len)
        self.flat_size = 256 * (seq_len // 4)  # approx after two stride=2
        
        self.mu = nn.Linear(self.flat_size, latent_dim)
        self.logvar = nn.Linear(self.flat_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flat_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, seq_len // 4)),
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, embed_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU()
        )
        
        self.output = nn.Linear(embed_dim, vocab_size)  # for reconstruction
        
    def encode(self, x):
        # x: (batch, seq_len) token indices
        x = self.embed(x)                    # → (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)               # → (batch, embed_dim, seq_len) for Conv1d
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    
class HybridConvVAE(nn.Module):
    def __init__(self, text_vocab_size, audio_mfcc_dim=13*100, latent_dim=64):
        super().__init__()
        # Lyrics branch (CNN)
        self.text_cnn = ...  # as above
        
        # Audio branch (CNN on MFCC spectrogram-like)
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(audio_mfcc_dim, 128, 5, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, 5, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Fusion
        self.fusion = nn.Linear(text_flat + audio_flat, latent_dim*2)
        self.mu = nn.Linear(latent_dim*2, latent_dim)
        self.logvar = nn.Linear(latent_dim*2, latent_dim)
        
        # Decoder would reconstruct both branches (multi-task)    


class BetaVAE(VAE):
    """
    Beta-VAE for disentangled representations (extends base VAE)
    Beta >1 encourages disentanglement
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, beta=4.0):
        super(BetaVAE, self).__init__(input_dim, hidden_dim, latent_dim)
        self.beta = beta

def train_beta_vae(features, hidden_dim=128, latent_dim=10, beta=4.0, epochs=100, batch_size=16, lr=0.001):
    dataset = TextDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BetaVAE(features.shape[1], hidden_dim, latent_dim, beta)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            recon_loss = nn.MSELoss()(recon, batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + model.beta * (kl_loss / batch.size(0))
            loss.backward()
            optimizer.step()
    
    return model

# Reconstruction example function
def reconstruct_sample(model, original_feat):
    with torch.no_grad():
        recon, _, _ = model(torch.from_numpy(original_feat).unsqueeze(0))
        return recon.numpy().squeeze()       

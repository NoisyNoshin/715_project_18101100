from src.dataset import TextDataset
from scipy.cluster.vq import kmeans, vq
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.vq import kmeans, vq
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.cluster import SpectralClustering

def kmeans_clustering(X, k=5):
    """K-Means clustering using scikit-learn (recommended over scipy version)"""
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return labels

def agglomerative_clustering(X, k=5):
    """Agglomerative (hierarchical) clustering with ward linkage"""
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = model.fit_predict(X)
    return labels

def dbscan_clustering(X, eps=0.5, min_samples=5):
    """DBSCAN clustering (density-based) - may produce noise points (-1 label)"""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels

# Keep your original functions for compatibility / baseline
def perform_clustering(latent, k=5):
    """Original scipy-based K-Means (can be replaced later)"""
    centroids, distortion = kmeans(latent, k)
    labels, _ = vq(latent, centroids)
    return labels

def pca_baseline(features, latent_dim, k=5):
    """PCA + K-Means baseline (unchanged)"""
    U, S, V = np.linalg.svd(features, full_matrices=False)
    pca_features = U[:, :latent_dim] * S[:latent_dim]
    centroids, _ = kmeans(pca_features, k)
    labels, _ = vq(pca_features, centroids)
    return labels, pca_features

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Decoder symmetric but not used for clustering

def train_ae(features, latent_dim=10, epochs=100):
    model = AutoEncoder(features.shape[1], 128, latent_dim)
    optimizer = optim.Adam(model.parameters())
    loader = DataLoader(TextDataset(features), batch_size=16, shuffle=True)
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            latent = model.encoder(batch)
            # Reconstruction loss etc. (simplified)
            loss = nn.MSELoss()(model.decoder(latent), batch)  # assume decoder added
            loss.backward()
            optimizer.step()
    return model.encoder

def ae_baseline(features, latent_dim, k=5):
    ae_encoder = train_ae(features, latent_dim)
    with torch.no_grad():
        ae_features = ae_encoder(torch.from_numpy(features)).numpy()
    return kmeans_clustering(ae_features, k), ae_features

def spectral_clustering(X, k=5):
    model = SpectralClustering(n_clusters=k, affinity='rbf', random_state=42)
    labels = model.fit_predict(X)
    return labels

def direct_spectral_baseline(features, k=5):
    labels = spectral_clustering(features, k=k)
    return labels, features

from scipy.cluster.vq import kmeans, vq
import numpy as np

def perform_clustering(latent, k=5):
    centroids, distortion = kmeans(latent, k)
    labels, _ = vq(latent, centroids)
    return labels

def pca_baseline(features, latent_dim, k=5):
    U, S, V = np.linalg.svd(features, full_matrices=False)
    pca_features = U[:, :latent_dim] * S[:latent_dim]
    centroids, _ = kmeans(pca_features, k)
    labels, _ = vq(pca_features, centroids)
    return labels, pca_features
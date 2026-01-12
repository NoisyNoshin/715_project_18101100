import numpy as np
from scipy.special import comb  # for binomial coefficient in Rand Index


def silhouette_score(X, labels):
    """
    Silhouette Score
    Formula: s(i) = (b(i) - a(i)) / max(a(i), b(i))
    Range: [-1, 1] — Higher is better
    """
    n = X.shape[0]
    unique_labels = np.unique(labels)
    sil = np.zeros(n)
    
    for i in range(n):
        # a(i): mean distance to other points in same cluster
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) > 1:
            distances = np.linalg.norm(same_cluster - X[i], axis=1)
            a = np.sum(distances) / (len(same_cluster) - 1)  # exclude self
        else:
            a = 0.0
        
        # b(i): min mean distance to points in any other cluster
        b_min = np.inf
        for lbl in unique_labels:
            if lbl != labels[i]:
                other_cluster = X[labels == lbl]
                if len(other_cluster) > 0:
                    mean_dist = np.mean(np.linalg.norm(other_cluster - X[i], axis=1))
                    if mean_dist < b_min:
                        b_min = mean_dist
        
        # Handle edge case where point is alone or no other clusters
        if b_min == np.inf:
            sil[i] = 0.0
        else:
            sil[i] = (b_min - a) / max(a, b_min) if max(a, b_min) > 0 else 0.0
    
    return np.mean(sil)


def calinski_harabasz_score(X, labels):
    """
    Calinski-Harabasz Index (Variance Ratio Criterion)
    Formula: CH = trace(B_k)/(k-1)  /  trace(W_k)/(n-k)
    Higher is better
    """
    n = X.shape[0]
    k = len(np.unique(labels))
    
    if k <= 1 or n <= k:
        return 0.0
    
    global_mean = np.mean(X, axis=0)
    B = 0.0  # between-cluster dispersion
    W = 0.0  # within-cluster dispersion
    
    for lbl in np.unique(labels):
        cluster_points = X[labels == lbl]
        if len(cluster_points) == 0:
            continue
        cluster_mean = np.mean(cluster_points, axis=0)
        # Between
        B += len(cluster_points) * np.sum((cluster_mean - global_mean) ** 2)
        # Within
        W += np.sum(np.sum((cluster_points - cluster_mean) ** 2, axis=1))
    
    if W == 0:
        return 0.0
    
    return (B / (k - 1)) / (W / (n - k))


def davies_bouldin_score(X, labels):
    """
    Davies-Bouldin Index
    Formula: DB = (1/k) * Σ max_{j≠i} ( (σ_i + σ_j) / d_ij )
    Lower is better
    """
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    if k <= 1:
        return 0.0
    
    centroids = np.array([np.mean(X[labels == lbl], axis=0) for lbl in unique_labels])
    sigmas = np.zeros(k)  # intra-cluster distances (average to centroid)
    
    for i, lbl in enumerate(unique_labels):
        cluster_points = X[labels == lbl]
        if len(cluster_points) > 0:
            sigmas[i] = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))
    
    db = 0.0
    for i in range(k):
        max_ratio = 0.0
        for j in range(k):
            if i != j:
                if sigmas[i] + sigmas[j] == 0:
                    ratio = 0.0
                else:
                    dist_centroids = np.linalg.norm(centroids[i] - centroids[j])
                    ratio = (sigmas[i] + sigmas[j]) / dist_centroids if dist_centroids > 0 else 0.0
                if ratio > max_ratio:
                    max_ratio = ratio
        db += max_ratio
    
    return db / k


def adjusted_rand_score(labels_true, labels_pred):
    """
    Adjusted Rand Index (ARI)
    Requires ground truth labels
    Range: [-1, 1] — Higher is better (1 = perfect, 0 = random)
    """
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true == c) & (labels_pred == k))
    
    sum_comb_c = np.sum([comb(n_c, 2) for n_c in np.sum(contingency, axis=1)])
    sum_comb_k = np.sum([comb(n_k, 2) for n_k in np.sum(contingency, axis=0)])
    sum_comb = np.sum([comb(n_ij, 2) for n_ij in contingency.flatten()])
    
    expected_index = (sum_comb_c * sum_comb_k) / comb(len(labels_true), 2)
    max_index = 0.5 * (sum_comb_c + sum_comb_k)
    
    if max_index == expected_index:
        return 0.0
    
    ari = (sum_comb - expected_index) / (max_index - expected_index)
    return ari


def normalized_mutual_info_score(labels_true, labels_pred):
    """
    Normalized Mutual Information (NMI)
    Requires ground truth labels
    Range: [0, 1] — Higher is better
    """
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    
    contingency = np.zeros((len(classes), len(clusters)), dtype=np.float64)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true == c) & (labels_pred == k))
    
    pi = np.sum(contingency, axis=1) / len(labels_true)
    pj = np.sum(contingency, axis=0) / len(labels_true)
    
    pij = contingency / len(labels_true)
    
    # Mutual Information I(U;V)
    mi = 0.0
    for i in range(len(classes)):
        for j in range(len(clusters)):
            if pij[i, j] > 0:
                mi += pij[i, j] * np.log(pij[i, j] / (pi[i] * pj[j]))
    
    # Entropies H(U), H(V)
    hu = -np.sum(pi * np.log(pi + 1e-15))
    hv = -np.sum(pj * np.log(pj + 1e-15))
    
    if hu == 0 or hv == 0:
        return 0.0
    
    return (2.0 * mi) / (hu + hv)


def cluster_purity(labels_true, labels_pred):
    """
    Cluster Purity
    Requires ground truth labels
    Range: [0, 1] — Higher is better
    Formula: (1/n) * Σ_k max_j |c_k ∩ t_j|
    """
    n = len(labels_true)
    unique_pred = np.unique(labels_pred)
    purity = 0.0
    
    for cluster in unique_pred:
        mask = (labels_pred == cluster)
        if np.sum(mask) == 0:
            continue
        cluster_labels = labels_true[mask]
        # Most frequent true label in this cluster
        dominant_count = np.max(np.bincount(cluster_labels))
        purity += dominant_count
    
    return purity / n

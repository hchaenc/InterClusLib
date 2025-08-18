import pandas as pd
import numpy as np
from collections import defaultdict
from interClusLib.metric import SIMILARITY_FUNCTIONS, DISTANCE_FUNCTIONS

def silhouette_score(data, labels, metric, centers=None) -> float:
    """
    Computes the silhouette score for a given clustering solution with vectorized distance calculations.

    Parameters
    ----------
    data : np.ndarray
        Shape (N, ...) data array. E.g. for interval data, (N, d, 2).
        If it's standard numeric data, shape (N, d).
    labels : np.ndarray
        Shape (N,). The cluster label for each sample.
    metric : str or callable
        A function (a, b) -> float, returning the distance between two samples.
        If is_similarity=True, we interpret 'distance_func' as similarity 
        and do distance = 1 - similarity inside. But typically you just
        pass in a true distance function.
    centers : np.ndarray, optional
        Cluster centers (not used in silhouette but kept for compatibility).

    Returns
    -------
    float
        The silhouette score in [-1, 1]. Usually > 0 indicates a reasonable 
        structure, closer to 1 is better.

    Example
    -------
        # Suppose you used an Interval K-Means or hierarchical clustering 
        # giving you 'labels' for data. Then:
        score = silhouette_score(data, labels, metric='euclidean')
        print("Silhouette:", score)
    """
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length.")

    n_samples = len(data)
    
    # Handle trivial cases
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1:
        return 0.0  # Only one cluster
    
    if n_samples <= 1:
        return 0.0  # Only one sample

    # Get distance function
    if isinstance(metric, str):
        if metric in SIMILARITY_FUNCTIONS:
            distance_func = SIMILARITY_FUNCTIONS[metric]
            is_sim = True
        elif metric in DISTANCE_FUNCTIONS:
            distance_func = DISTANCE_FUNCTIONS[metric]
            is_sim = False
        else:
            valid_metrics = ", ".join(list(SIMILARITY_FUNCTIONS.keys()) + list(DISTANCE_FUNCTIONS.keys()))
            raise ValueError(f"Invalid metric '{metric}'. Available options: {valid_metrics}")
    else:
        distance_func = metric
        is_sim = False  # Assume distance by default for custom functions

    # Try vectorized distance computation first
    try:
        return _silhouette_score_vectorized(data, labels, distance_func, is_sim, unique_labels)
    except Exception as e:
        print(f"Vectorized computation failed ({e}), falling back to pairwise...")
        return _silhouette_score_pairwise(data, labels, distance_func, is_sim, unique_labels)


def _silhouette_score_vectorized(data, labels, distance_func, is_sim, unique_labels):
    """
    Vectorized silhouette computation using full distance matrix.
    """
    n_samples = len(data)
    
    # Compute full distance matrix
    distance_matrix = distance_func(data, data)
    
    if is_sim:
        distance_matrix = 1.0 - distance_matrix
    
    # Ensure diagonal is zero and matrix is symmetric
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
    
    # Build cluster membership arrays
    cluster_masks = {}
    cluster_sizes = {}
    
    for label in unique_labels:
        mask = labels == label
        cluster_masks[label] = mask
        cluster_sizes[label] = np.sum(mask)
    
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        current_label = labels[i]
        current_mask = cluster_masks[current_label]
        
        # Compute a(i) - average distance within cluster
        if cluster_sizes[current_label] > 1:
            # Get distances to same cluster (excluding self)
            same_cluster_dists = distance_matrix[i, current_mask]
            # Remove self-distance (should be 0)
            same_cluster_dists = same_cluster_dists[same_cluster_dists > 0]
            a_i = np.mean(same_cluster_dists) if len(same_cluster_dists) > 0 else 0.0
        else:
            a_i = 0.0
        
        # Compute b(i) - minimum average distance to other clusters
        b_candidates = []
        for other_label in unique_labels:
            if other_label == current_label:
                continue
            
            other_mask = cluster_masks[other_label]
            if cluster_sizes[other_label] > 0:
                other_cluster_dists = distance_matrix[i, other_mask]
                avg_dist = np.mean(other_cluster_dists)
                b_candidates.append(avg_dist)
        
        b_i = min(b_candidates) if b_candidates else 0.0
        
        # Compute silhouette value
        max_ab = max(a_i, b_i)
        if max_ab > 1e-15:
            silhouette_vals[i] = (b_i - a_i) / max_ab
        else:
            silhouette_vals[i] = 0.0
    
    return np.mean(silhouette_vals)


def _silhouette_score_pairwise(data, labels, distance_func, is_sim, unique_labels):
    """
    Fallback pairwise computation when vectorized method fails.
    """
    n_samples = len(data)
    
    # Build cluster indexing
    cluster_indices = {}
    for label in unique_labels:
        cluster_indices[label] = np.where(labels == label)[0]
    
    silhouette_vals = np.zeros(n_samples)
    
    for i in range(n_samples):
        current_label = labels[i]
        current_cluster_indices = cluster_indices[current_label]
        
        # Compute a(i) - average intra-cluster distance
        if len(current_cluster_indices) > 1:
            intra_distances = []
            for j in current_cluster_indices:
                if j != i:
                    dist = distance_func(data[i], data[j])
                    if is_sim:
                        dist = 1.0 - dist
                    intra_distances.append(dist)
            a_i = np.mean(intra_distances) if intra_distances else 0.0
        else:
            a_i = 0.0
        
        # Compute b(i) - minimum average inter-cluster distance
        b_candidates = []
        for other_label in unique_labels:
            if other_label == current_label:
                continue
            
            other_cluster_indices = cluster_indices[other_label]
            if len(other_cluster_indices) > 0:
                inter_distances = []
                for j in other_cluster_indices:
                    dist = distance_func(data[i], data[j])
                    if is_sim:
                        dist = 1.0 - dist
                    inter_distances.append(dist)
                
                avg_inter_dist = np.mean(inter_distances)
                b_candidates.append(avg_inter_dist)
        
        b_i = min(b_candidates) if b_candidates else 0.0
        
        # Compute silhouette value
        max_ab = max(a_i, b_i)
        if max_ab > 1e-15:
            silhouette_vals[i] = (b_i - a_i) / max_ab
        else:
            silhouette_vals[i] = 0.0
    
    return np.mean(silhouette_vals)
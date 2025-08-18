import numpy as np
from collections import defaultdict
from interClusLib.metric import SIMILARITY_FUNCTIONS, DISTANCE_FUNCTIONS

def calinski_harabasz_index(
        data: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        metric,
    ) -> float:
    """
    Compute the Calinski-Harabasz (CH) Index for a given clustering with vectorized distance calculations.

    Parameters
    ----------
    data : np.ndarray
        Shape (N, ...). For interval data => (N, n_dims,2).
    labels : np.ndarray
        Shape (N,). The cluster label for each sample.
    centers : np.ndarray
        Shape (k, ...). The cluster centers (for numeric data or intervals).
        E.g. for interval data => shape(k, n_dims,2).
    metric : str or callable
        Distance/similarity function name or callable.

    Returns
    -------
    float
        The Calinski-Harabasz index. Larger => better cluster separation.
        If only one cluster or other degenerate cases => returns 0.

    Notes
    -----
    CH = (B / (k - 1)) / (W / (n - k)),
        where B is "between-cluster scatter", W is "within-cluster scatter".
    For data shape => (N, d,2) in interval scenario:
        B = sum_c( n_c * dist^2(centroid_c, centroid_global) )
        W = sum_c sum_{x in c}( dist^2(x, centroid_c) )

    'centroid_global' = average of all data => shape(d,2). 
    'centroid_c' is from 'centers[c]'.
    """
    # Check shapes
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length.")

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
        is_sim = False

    # Basic definitions
    n_samples = len(data)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    if k < 2 or n_samples == k:
        # CH index not well-defined if only 1 cluster or each sample in own cluster
        return 0.0

    # Try vectorized computation first
    try:
        return _calinski_harabasz_vectorized(data, labels, centers, distance_func, is_sim, unique_labels, n_samples, k)
    except Exception as e:
        print(f"Vectorized computation failed ({e}), falling back to pairwise...")
        return _calinski_harabasz_pairwise(data, labels, centers, distance_func, is_sim, unique_labels, n_samples, k)


def _calinski_harabasz_vectorized(data, labels, centers, distance_func, is_sim, unique_labels, n_samples, k):
    """
    Vectorized Calinski-Harabasz computation using distance matrices.
    """
    # Compute global centroid
    global_center = np.mean(data, axis=0)
    
    # Build cluster membership masks
    cluster_masks = {}
    cluster_sizes = {}
    
    for label in unique_labels:
        mask = labels == label
        cluster_masks[label] = mask
        cluster_sizes[label] = np.sum(mask)
    
    # Compute B (between-cluster scatter) vectorized
    try:
        # Try to compute all center-to-global distances at once
        center_to_global_dists = distance_func(centers, global_center[None, :])
        
        if is_sim:
            center_to_global_dists = 1.0 - center_to_global_dists
        
        # Handle different output shapes
        if center_to_global_dists.ndim == 1:
            center_dists = center_to_global_dists
        elif center_to_global_dists.ndim == 2:
            center_dists = center_to_global_dists[:, 0]  # Take first column if matrix
        else:
            raise ValueError("Unexpected distance matrix shape")
        
        # Compute B using vectorized operations
        cluster_sizes_array = np.array([cluster_sizes[label] for label in unique_labels])
        B = np.sum(cluster_sizes_array * (center_dists ** 2))
        
    except Exception:
        # Fallback to pairwise for B computation
        B = 0.0
        for i, label in enumerate(unique_labels):
            n_c = cluster_sizes[label]
            if n_c == 0:
                continue
            dist_c = distance_func(centers[i], global_center)
            if is_sim:
                dist_c = 1.0 - dist_c
            B += n_c * (dist_c ** 2)
    
    # Compute W (within-cluster scatter) vectorized
    try:
        # Try to compute all data-to-center distances at once
        data_to_centers_dists = distance_func(data, centers)
        
        if is_sim:
            data_to_centers_dists = 1.0 - data_to_centers_dists
        
        # Extract distances for each sample to its assigned center
        W = 0.0
        for i, label in enumerate(unique_labels):
            mask = cluster_masks[label]
            if np.any(mask):
                # Get distances from samples in this cluster to their center
                if data_to_centers_dists.ndim == 2:
                    cluster_dists = data_to_centers_dists[mask, i]
                else:
                    # Handle 1D case
                    cluster_dists = data_to_centers_dists[mask]
                
                W += np.sum(cluster_dists ** 2)
                
    except Exception:
        # Fallback to pairwise for W computation
        W = 0.0
        for i, label in enumerate(unique_labels):
            center = centers[i]
            mask = cluster_masks[label]
            cluster_data = data[mask]
            
            for sample in cluster_data:
                dist_ic = distance_func(sample, center)
                if is_sim:
                    dist_ic = 1.0 - dist_ic
                W += dist_ic ** 2
    
    # Compute CH index
    if abs(W) < 1e-15:
        return 0.0 if B == 0 else float('inf')
    
    CH = (B / (k - 1)) / (W / (n_samples - k))
    return CH


def _calinski_harabasz_pairwise(data, labels, centers, distance_func, is_sim, unique_labels, n_samples, k):
    """
    Fallback pairwise computation when vectorized method fails.
    """
    # Compute global centroid
    global_center = np.mean(data, axis=0)
    
    # Build cluster map
    cluster_map = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_map[lab].append(i)
    
    # Distance function wrapper
    def dist(a, b):
        val = distance_func(a, b)
        if is_sim:
            return 1.0 - val
        else:
            return val
    
    # Compute B: sum_{c} [ n_c * dist^2(centroid_c, global_center) ]
    B = 0.0
    for i, c in enumerate(unique_labels):
        n_c = len(cluster_map[c])
        if n_c == 0:
            continue
        dist_c = dist(centers[i], global_center)
        B += n_c * (dist_c ** 2)
    
    # Compute W: sum_{c} sum_{ x in c } dist^2( x, centroid_c )
    W = 0.0
    for i, c in enumerate(unique_labels):
        centroid_c = centers[i]
        idxs = cluster_map[c]
        for idx in idxs:
            dist_ic = dist(data[idx], centroid_c)
            W += dist_ic ** 2
    
    # Compute CH index
    if abs(W) < 1e-15:
        return 0.0 if B == 0 else float('inf')
    
    CH = (B / (k - 1)) / (W / (n_samples - k))
    return CH
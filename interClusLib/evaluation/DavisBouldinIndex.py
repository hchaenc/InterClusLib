from collections import defaultdict
import numpy as np
from interClusLib.metric import SIMILARITY_FUNCTIONS, DISTANCE_FUNCTIONS

def davies_bouldin_index(
        data: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        metric) -> float:
    """
    Compute the Davies-Bouldin Index for a clustering solution with vectorized distance calculations.

    Parameters
    ----------
    data : np.ndarray
        Shape (N, ...) dataset. E.g. for interval data, (N, n_dims, 2).
    labels : np.ndarray
        Shape (N,). The cluster label for each sample.
    centers : np.ndarray
        Shape (k, ...). The center (prototype) of each cluster.
        For interval data, might be shape (k, n_dims, 2).
    metric : str or callable
        Distance/similarity function name or callable.

    Returns
    -------
    float
        The Davies-Bouldin Index. Smaller => better cluster separation.

    Example
    -------
    # Suppose we used IntervalKMeans => got labels, centers, etc.
    # data shape: (N, n_dims, 2)
    dbi = davies_bouldin_index(data, labels, centers, metric='euclidean')
    print("Davies-Bouldin =", dbi)
    """
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

    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k < 2:
        return 0.0

    # Try vectorized computation first
    try:
        return _davies_bouldin_vectorized(data, labels, centers, distance_func, is_sim, unique_labels)
    except Exception as e:
        print(f"Vectorized computation failed ({e}), falling back to pairwise...")
        return _davies_bouldin_pairwise(data, labels, centers, distance_func, is_sim, unique_labels)


def _davies_bouldin_vectorized(data, labels, centers, distance_func, is_sim, unique_labels):
    """
    Vectorized Davies-Bouldin Index computation.
    """
    k = len(unique_labels)
    
    # Build cluster membership
    cluster_masks = {}
    cluster_indices = {}
    
    for label in unique_labels:
        mask = labels == label
        indices = np.where(mask)[0]
        cluster_masks[label] = mask
        cluster_indices[label] = indices
    
    # Filter out empty clusters
    non_empty_labels = [lab for lab in unique_labels if len(cluster_indices[lab]) > 0]
    
    if len(non_empty_labels) < 2:
        return 0.0
    
    # 1) Compute S_i (average intra-cluster distances) vectorized
    S = {}
    
    try:
        # Try vectorized computation of data-to-centers distances
        data_to_centers = distance_func(data, centers)
        
        if is_sim:
            data_to_centers = 1.0 - data_to_centers
        
        # Extract distances for each cluster
        for i, label in enumerate(non_empty_labels):
            mask = cluster_masks[label]
            if data_to_centers.ndim == 2:
                cluster_distances = data_to_centers[mask, i]
            else:
                # Handle 1D case
                cluster_distances = data_to_centers[mask]
            
            S[label] = np.mean(cluster_distances)
            
    except Exception:
        # Fallback to pairwise for S computation
        for i, label in enumerate(non_empty_labels):
            indices = cluster_indices[label]
            if len(indices) == 0:
                S[label] = 0.0
                continue
            
            distances = []
            for idx in indices:
                dist_val = distance_func(data[idx], centers[i])
                if is_sim:
                    dist_val = 1.0 - dist_val
                distances.append(dist_val)
            
            S[label] = np.mean(distances)
    
    # 2) Compute center-to-center distances vectorized
    center_distances = {}
    
    try:
        # Try vectorized computation of center-to-center distances
        center_to_center = distance_func(centers, centers)
        
        if is_sim:
            center_to_center = 1.0 - center_to_center
        
        # Extract pairwise center distances
        for i, label_i in enumerate(non_empty_labels):
            for j, label_j in enumerate(non_empty_labels):
                if i != j:
                    if center_to_center.ndim == 2:
                        center_distances[(label_i, label_j)] = center_to_center[i, j]
                    else:
                        # Handle 1D case - should not happen for center-to-center
                        center_distances[(label_i, label_j)] = center_to_center[i] if i < len(center_to_center) else 0.0
                        
    except Exception:
        # Fallback to pairwise for center distances
        for i, label_i in enumerate(non_empty_labels):
            for j, label_j in enumerate(non_empty_labels):
                if i != j:
                    dist_val = distance_func(centers[i], centers[j])
                    if is_sim:
                        dist_val = 1.0 - dist_val
                    center_distances[(label_i, label_j)] = dist_val
    
    # 3) Compute Davies-Bouldin Index
    DB_sum = 0.0
    valid_clusters_count = 0
    
    for label_i in non_empty_labels:
        if len(cluster_indices[label_i]) == 0:
            continue
        
        max_R_ij = -1e15
        
        for label_j in non_empty_labels:
            if label_i == label_j or len(cluster_indices[label_j]) == 0:
                continue
            
            d_ij = center_distances.get((label_i, label_j), None)
            if d_ij is None or abs(d_ij) < 1e-15:
                continue
            
            R_ij = (S[label_i] + S[label_j]) / d_ij
            if R_ij > max_R_ij:
                max_R_ij = R_ij
        
        if max_R_ij > -1e15:
            DB_sum += max_R_ij
            valid_clusters_count += 1
    
    if valid_clusters_count < 1:
        return 0.0
    
    dbi = DB_sum / valid_clusters_count
    return dbi


def _davies_bouldin_pairwise(data, labels, centers, distance_func, is_sim, unique_labels):
    """
    Fallback pairwise computation when vectorized method fails.
    """
    # Distance function wrapper
    def dist(x, y):
        val = distance_func(x, y)
        if is_sim:
            return 1.0 - val
        else:
            return val
    
    # Build cluster map
    cluster_map = {}
    for c in unique_labels:
        cluster_map[c] = []
    for i, lab in enumerate(labels):
        cluster_map[lab].append(i)
    
    k = len(unique_labels)
    
    # Compute S_i = average distance from each sample in cluster i to center i
    S = {}
    for i, c in enumerate(unique_labels):
        idxs = cluster_map[c]
        if len(idxs) < 1:
            S[c] = 0.0
            continue
        
        dist_sum = 0.0
        for idx in idxs:
            dist_sum += dist(data[idx], centers[i])
        S[c] = dist_sum / len(idxs)
    
    # Compute center-to-center distances
    center_dist = {}
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i == j:
                continue
            d_ij = dist(centers[i], centers[j])
            center_dist[(label_i, label_j)] = d_ij
    
    # Compute Davies-Bouldin Index
    DB_sum = 0.0
    valid_clusters_count = 0
    
    for i, label_i in enumerate(unique_labels):
        if len(cluster_map[label_i]) == 0:
            continue
        
        max_ij = -1e15
        for j, label_j in enumerate(unique_labels):
            if j == i or len(cluster_map[label_j]) == 0:
                continue
            
            d_ij = center_dist.get((label_i, label_j), None)
            if (d_ij is None) or (abs(d_ij) < 1e-15):
                continue
            
            R_ij = (S[label_i] + S[label_j]) / d_ij
            if R_ij > max_ij:
                max_ij = R_ij
        
        if max_ij > -1e15:
            DB_sum += max_ij
            valid_clusters_count += 1
    
    if valid_clusters_count < 1:
        return 0.0
    
    dbi = DB_sum / valid_clusters_count
    return dbi
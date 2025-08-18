import numpy as np
from collections import defaultdict
from interClusLib.metric import SIMILARITY_FUNCTIONS, DISTANCE_FUNCTIONS

def dunn_index(data: np.ndarray,
                labels: np.ndarray,
                metric, centers=None) -> float:
    """
    Computes the Dunn Index for clustering with vectorized distance calculations.
    
    Parameters
    ----------
    data : np.ndarray
        Shape (N, ...) dataset. For interval data, could be (N, n_dims, 2).
        For normal numeric data, (N,d).
    labels : np.ndarray
        Shape (N,). Each sample's cluster label.
    metric : str or callable
        Distance/similarity function name or callable.
    centers : np.ndarray, optional
        Cluster centers (not used in Dunn but kept for compatibility).
    
    Returns
    -------
    dunn_value : float
        Dunn index. Larger => better separation between clusters & small intracluster diameter.
        If any cluster has <2 points, that cluster's diameter = 0.
        If a cluster is empty, it's skipped in the min inter-cluster distance loop.

    Example
    -------
        # Suppose you used IntervalKMeans => got labels
        # and have data shape (N, n_dims,2)
        dunn = dunn_index(data, labels, metric='euclidean')
        print("Dunn Index =", dunn)
    """
    if len(data) != len(labels):
        raise ValueError("data and labels must have same length.")
    
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
    
    # Try vectorized computation first
    try:
        return _dunn_index_vectorized(data, labels, distance_func, is_sim, unique_labels)
    except Exception as e:
        print(f"Vectorized computation failed ({e}), falling back to pairwise...")
        return _dunn_index_pairwise(data, labels, distance_func, is_sim, unique_labels)


def _dunn_index_vectorized(data, labels, distance_func, is_sim, unique_labels):
    """
    Vectorized Dunn Index computation using distance matrices.
    """
    n_samples = len(data)
    
    # Compute full distance matrix
    distance_matrix = distance_func(data, data)
    
    if is_sim:
        distance_matrix = 1.0 - distance_matrix
    
    # Ensure diagonal is zero and matrix is symmetric
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
    
    # Build cluster membership masks
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
    
    # 1) Compute cluster diameters (max intra-cluster distance)
    cluster_diameters = {}
    
    for label in non_empty_labels:
        indices = cluster_indices[label]
        if len(indices) < 2:
            cluster_diameters[label] = 0.0
            continue
        
        # Extract submatrix for this cluster
        cluster_distances = distance_matrix[np.ix_(indices, indices)]
        # Get maximum distance (diameter)
        cluster_diameters[label] = np.max(cluster_distances)
    
    # 2) Compute minimum inter-cluster distance
    min_inter_dist = float('inf')
    
    for i in range(len(non_empty_labels)):
        for j in range(i + 1, len(non_empty_labels)):
            label1 = non_empty_labels[i]
            label2 = non_empty_labels[j]
            
            indices1 = cluster_indices[label1]
            indices2 = cluster_indices[label2]
            
            # Extract inter-cluster distances
            inter_distances = distance_matrix[np.ix_(indices1, indices2)]
            local_min = np.min(inter_distances)
            
            if local_min < min_inter_dist:
                min_inter_dist = local_min
    
    # 3) Compute Dunn Index
    if min_inter_dist == float('inf'):
        return 0.0
    
    max_diam = max(cluster_diameters.values()) if cluster_diameters else 0.0
    
    if max_diam < 1e-15:
        if min_inter_dist > 0:
            return float('inf')
        else:
            return 0.0
    
    dunn = min_inter_dist / max_diam
    return dunn


def _dunn_index_pairwise(data, labels, distance_func, is_sim, unique_labels):
    """
    Fallback pairwise computation when vectorized method fails.
    """
    # Distance function wrapper
    def dist(a, b):
        val = distance_func(a, b)
        if is_sim:
            return 1.0 - val
        else:
            return val
    
    # Build cluster map
    cluster_map = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_map[lab].append(i)
    
    # Compute cluster diameters
    cluster_diameters = {}
    for lab in unique_labels:
        idxs = cluster_map[lab]
        if len(idxs) < 2:
            cluster_diameters[lab] = 0.0
            continue
        
        # Max distance among points in cluster
        max_dist = 0.0
        for i_idx in range(len(idxs)):
            for j_idx in range(i_idx + 1, len(idxs)):
                dist_ij = dist(data[idxs[i_idx]], data[idxs[j_idx]])
                if dist_ij > max_dist:
                    max_dist = dist_ij
        cluster_diameters[lab] = max_dist
    
    # Compute inter-cluster distance
    labs_list = [lab for lab in unique_labels if len(cluster_map[lab]) > 0]
    if len(labs_list) < 2:
        return 0.0
    
    min_inter_dist = float('inf')
    for i in range(len(labs_list)):
        for j in range(i + 1, len(labs_list)):
            c1 = labs_list[i]
            c2 = labs_list[j]
            
            idxs1 = cluster_map[c1]
            idxs2 = cluster_map[c2]
            local_min = float('inf')
            
            for idx_p in idxs1:
                for idx_q in idxs2:
                    dist_pq = dist(data[idx_p], data[idx_q])
                    if dist_pq < local_min:
                        local_min = dist_pq
            
            if local_min < min_inter_dist:
                min_inter_dist = local_min
    
    # Compute Dunn Index
    if min_inter_dist == float('inf'):
        return 0.0
    
    max_diam = max(cluster_diameters.values()) if cluster_diameters else 0.0
    if max_diam < 1e-15:
        if min_inter_dist > 0:
            return float('inf')
        else:
            return 0.0
    
    dunn = min_inter_dist / max_diam
    return dunn
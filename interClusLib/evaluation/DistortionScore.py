import pandas as pd
import numpy as np
from collections import defaultdict
from interClusLib.metric import SIMILARITY_FUNCTIONS, DISTANCE_FUNCTIONS

def distortion_score(data, labels, centers, metric) -> float:
    """
    Computes the distortion score for a given clustering solution with vectorized distance calculations.

    Parameters
    ----------
    data : np.ndarray
        Shape (N, ...) data array. E.g. for interval data, (N, d, 2).
        If it's standard numeric data, shape (N, d).
    labels : np.ndarray
        Shape (N,). The cluster label for each sample.
    centers : np.ndarray
        Shape (k, ...) array of cluster centers. For interval data, (k, d, 2).
    metric : str or callable
        Distance/similarity function name or callable.

    Returns
    -------
    float
        The distortion score. Lower values indicate better clustering.
    """
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

    n_samples = len(data)
    if n_samples == 0:
        return 0.0

    # Try vectorized computation first
    try:
        return _distortion_score_vectorized(data, labels, centers, distance_func, is_sim, n_samples)
    except Exception as e:
        print(f"Vectorized computation failed ({e}), falling back to pairwise...")
        return _distortion_score_pairwise(data, labels, centers, distance_func, is_sim, n_samples)


def _distortion_score_vectorized(data, labels, centers, distance_func, is_sim, n_samples):
    """
    Vectorized distortion score computation.
    """
    unique_labels = np.unique(labels)
    
    try:
        # Try to compute all data-to-centers distances at once
        data_to_centers = distance_func(data, centers)
        
        if is_sim:
            data_to_centers = 1.0 - data_to_centers
        
        # Extract distance for each sample to its assigned center
        total_distortion = 0.0
        
        if data_to_centers.ndim == 2:
            # Matrix case: (n_samples, n_centers)
            for i, label in enumerate(labels):
                if label < len(centers):  # Ensure valid center index
                    dist_val = data_to_centers[i, label]
                    total_distortion += dist_val ** 2
        else:
            # Vector case or other shapes - fall back to manual extraction
            for i, label in enumerate(labels):
                # Get distance from sample i to its assigned center
                center_idx = label
                if center_idx < len(centers):
                    # Compute distance for this specific pair
                    dist_val = distance_func(data[i], centers[center_idx])
                    if is_sim:
                        dist_val = 1.0 - dist_val
                    total_distortion += dist_val ** 2
        
        return total_distortion / n_samples
        
    except Exception:
        # If vectorized approach fails, compute cluster by cluster
        cluster_masks = {}
        for label in unique_labels:
            cluster_masks[label] = labels == label
        
        total_distortion = 0.0
        
        for cluster_id in unique_labels:
            if cluster_id >= len(centers):
                continue
                
            mask = cluster_masks[cluster_id]
            cluster_data = data[mask]
            center = centers[cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            # Try vectorized computation for this cluster
            try:
                cluster_distances = distance_func(cluster_data, center[None, :])
                
                if is_sim:
                    cluster_distances = 1.0 - cluster_distances
                
                # Handle different output shapes
                if cluster_distances.ndim == 2:
                    cluster_distances = cluster_distances[:, 0]
                
                cluster_distortion = np.sum(cluster_distances ** 2)
                total_distortion += cluster_distortion
                
            except Exception:
                # Fall back to pairwise for this cluster
                cluster_distortion = 0.0
                for sample in cluster_data:
                    dist_val = distance_func(sample, center)
                    if is_sim:
                        dist_val = 1.0 - dist_val
                    cluster_distortion += dist_val ** 2
                
                total_distortion += cluster_distortion
        
        return total_distortion / n_samples


def _distortion_score_pairwise(data, labels, centers, distance_func, is_sim, n_samples):
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
    
    # Compute distortion
    total_distortion = 0.0
    
    for cluster_id, center in enumerate(centers):
        if cluster_id not in cluster_map:
            continue
            
        # Get all points in this cluster
        cluster_points_idx = cluster_map[cluster_id]
        
        # Sum squared distances from points to center
        cluster_distortion = 0.0
        for idx in cluster_points_idx:
            # Square the distance
            cluster_distortion += dist(data[idx], center) ** 2
            
        total_distortion += cluster_distortion
    
    # Return average distortion per point
    return total_distortion / n_samples
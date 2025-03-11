import pandas as pd
import numpy as np
from collections import defaultdict
from interClusLib.metric import SIMILARITY_FUNCTIONS, DISTANCE_FUNCTIONS

def distortion_score(data, labels, centers, metric) -> float:
    """
    Computes the distortion score for a given clustering solution.

    Parameters
    ----------
    data : np.ndarray
        Shape (N, ...) data array. E.g. for interval data, (N, d, 2).
        If it's standard numeric data, shape (N, d).
    labels : np.ndarray
        Shape (N,). The cluster label for each sample.
    centers : np.ndarray
        Shape (k, ...) array of cluster centers. For interval data, (k, d, 2).
    metric : callable
        A function (a, b) -> float, returning the distance between two samples.
        If is_similarity=True, we interpret 'distance_func' as similarity 
        and do distance = 1 - similarity inside. But typically you just
        pass in a true distance function.

    Returns
    -------
    float
        The distortion score. Lower values indicate better clustering.
    """
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length.")

    if metric in SIMILARITY_FUNCTIONS:
        distance_func = SIMILARITY_FUNCTIONS[metric]
        is_sim = True
    elif metric in DISTANCE_FUNCTIONS:
        distance_func = DISTANCE_FUNCTIONS[metric]
        is_sim = False
    else:
        valid_metric = ", ".join(list(SIMILARITY_FUNCTIONS.keys()) + list(DISTANCE_FUNCTIONS.keys()))
        raise ValueError(f"Invalid metric '{metric}'. Available options: {valid_metric}")

    # If user gave us similarity, define a small wrapper
    def dist(a, b):
        val = distance_func(a, b)
        if is_sim:
            return 1.0 - val
        else:
            return val

    # Build cluster -> list of indices
    cluster_map = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_map[lab].append(i)
    
    # Compute distortion
    total_distortion = 0.0
    n_samples = len(data)
    
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
    return total_distortion / n_samples if n_samples > 0 else 0.0
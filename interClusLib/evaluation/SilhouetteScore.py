import pandas as pd
import numpy as np
from collections import defaultdict
from interClusLib.metric import SIMILARITY_FUNCTIONS, DISTANCE_FUNCTIONS

def silhouette_score(data, labels , metric, centers = None) -> float:
    """
    Computes the silhouette score for a given clustering solution.

    Parameters
    ----------
    data : np.ndarray
        Shape (N, ...) data array. E.g. for interval data, (N, d, 2).
        If it's standard numeric data, shape (N, d).
    labels : np.ndarray
        Shape (N,). The cluster label for each sample.
    metric : callable
        A function (a, b) -> float, returning the distance between two samples.
        If is_similarity=True, we interpret 'distance_func' as similarity 
        and do distance = 1 - similarity inside. But typically you just
        pass in a true distance function.

    Returns
    -------
    float
        The silhouette score in [-1, 1]. Usually > 0 indicates a reasonable 
        structure, closer to 1 is better.

    Example
    -------
        # Suppose you used an Interval K-Means or hierarchical clustering 
        # giving you 'labels' for data. Then:
        score = Evaluation.silhouette_score(data, labels, distance_func=hausdorff_dist)
        print("Silhouette:", score)
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

    n_samples = len(data)

    # If user gave us similarity, define a small wrapper
    def dist(a, b):
        val = distance_func(a, b)
        if is_sim:
            return 1.0 - val
        else:
            return val

    # Build cluster -> list of indices
    cluster_map = defaultdict(list)
    unique_labels = np.unique(labels)
    for i, lab in enumerate(labels):
        cluster_map[lab].append(i)

    # Silhouette for each sample
    silhouette_vals = np.zeros(n_samples, dtype=float)

    for i in range(n_samples):
        c_i = labels[i]
        # 1) a(i): avg distance to same cluster
        same_idxs = cluster_map[c_i]
        if len(same_idxs) > 1:
            dist_sum = 0.0
            count = 0
            for idx in same_idxs:
                if idx == i:
                    continue
                dist_sum += dist(data[i], data[idx])
                count += 1
            a_i = dist_sum / count if count>0 else 0.0
        else:
            # only one sample in this cluster => a(i)=0
            a_i = 0.0

        # 2) b(i): min over other clusters average distance
        b_candidates = []
        for clab in unique_labels:
            if clab == c_i:
                continue
            other_idxs = cluster_map[clab]
            if len(other_idxs) == 0:
                continue
            dist_sum2 = 0.0
            count2 = 0
            for idx in other_idxs:
                dist_sum2 += dist(data[i], data[idx])
                count2 += 1
            if count2>0:
                b_candidates.append(dist_sum2 / count2)

        if len(b_candidates) == 0:
            # degenerate: no other cluster
            b_i = 0.0
        else:
            b_i = min(b_candidates)

        # 3) silhouette
        m_ab = max(a_i, b_i)
        if m_ab > 1e-15:
            s_i = (b_i - a_i) / m_ab
        else:
            s_i = 0.0
        
        silhouette_vals[i] = s_i
    
    # 4) Return mean
    return np.mean(silhouette_vals)
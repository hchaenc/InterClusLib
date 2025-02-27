import numpy as np
from collections import defaultdict
from interClusLib.metric import MULTI_SIMILARITY_FUNCTIONS, MULTI_DISTANCE_FUNCTIONS

def dunn_index(data: np.ndarray,
                labels: np.ndarray,
                metric) -> float:
    """
    Computes the Dunn Index for an unlabeled clustering.
    
    Parameters
    ----------
    data : np.ndarray
        Shape (N, ...) dataset. For interval data, could be (N, n_dims, 2).
        For normal numeric data, (N,d).
    labels : np.ndarray
        Shape (N,). Each sample's cluster label.
    distance_func : callable
        Function (sampleA, sampleB) -> distance (float).
        E.g. for interval data: a Hausdorff function.
        For numeric data: Euclidean, Manhattan, etc.
    
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
        dunn = Evaluation.dunn_index(data, labels, distance_func=hausdorff_dist)
        print("Dunn Index =", dunn)
    """
    if len(data) != len(labels):
        raise ValueError("data and labels must have same length.")
    
    if metric in MULTI_SIMILARITY_FUNCTIONS:
        distance_func = MULTI_SIMILARITY_FUNCTIONS[metric]
        is_sim = True
    elif metric in MULTI_DISTANCE_FUNCTIONS:
        distance_func = MULTI_DISTANCE_FUNCTIONS[metric]
        is_sim = False
    else:
        valid_metric = ", ".join(list(MULTI_SIMILARITY_FUNCTIONS.keys()) + list(MULTI_DISTANCE_FUNCTIONS.keys()))
        raise ValueError(f"Invalid metric '{metric}'. Available options: {metric}")
    
    # If user gave us similarity, define a small wrapper
    def dist(a, b):
        val = distance_func(a, b)
        if is_sim:
            return 1.0 - val
        else:
            return val
    
    # 1) Build cluster -> sample indices
    cluster_map = defaultdict(list)
    unique_labels = np.unique(labels)
    for i, lab in enumerate(labels):
        cluster_map[lab].append(i)

    # 2) Compute 'diameter' for each cluster => max pairwise distance
    # If cluster has 0 or 1 sample => diameter=0
    cluster_diameters = {}
    for lab in unique_labels:
        idxs = cluster_map[lab]
        if len(idxs) < 2:
            cluster_diameters[lab] = 0.0
            continue
        # max distance among points in cluster
        max_dist = 0.0
        for i_idx in range(len(idxs)):
            for j_idx in range(i_idx+1, len(idxs)):
                dist_ij = dist(data[idxs[i_idx]], data[idxs[j_idx]])
                if dist_ij > max_dist:
                    max_dist = dist_ij
        cluster_diameters[lab] = max_dist
    
    # 3) Compute inter-cluster distance => min pairwise dist between any two clusters
    #   inter_cluster_dist(c1, c2) = min_{p in c1, q in c2} distance(p,q)
    # skip if cluster is empty
    labs_list = [lab for lab in unique_labels if len(cluster_map[lab])>0]
    if len(labs_list) < 2:
        # degenerate => only 1 cluster has data => Dunn=0
        return 0.0

    min_inter_dist = float('inf')
    for i in range(len(labs_list)):
        for j in range(i+1, len(labs_list)):
            c1 = labs_list[i]
            c2 = labs_list[j]
            # compute min distance of any pair across c1, c2
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

    # if for some reason all but one cluster are empty => min_inter_dist=inf => Dunn=0
    if min_inter_dist == float('inf'):
        return 0.0

    # 4) Dunn = min_inter_dist / max_intra_diam
    max_diam = max(cluster_diameters.values()) if cluster_diameters else 0.0
    if max_diam < 1e-15:
        if min_inter_dist>0:
            # means cluster diameter=0 but interdist>0 => big number
            # or define as 0 or some large => here we define a big number
            return float('inf')
        else:
            return 0.0
    dunn = min_inter_dist / max_diam
    return dunn
from collections import defaultdict
import numpy as np
from interClusLib.metric import SIMILARITY_FUNCTIONS, DISTANCE_FUNCTIONS

def davies_bouldin_index(
        data: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        metric,) -> float:
    """
    Compute the Davies-Bouldin Index for a clustering solution.

    Parameters
    ----------
    data : np.ndarray
        Shape (N, ...) dataset. E.g. for interval data, (N, n_dims, 2).
    labels : np.ndarray
        Shape (N,). The cluster label for each sample.
    centers : np.ndarray
        Shape (k, ...). The center (prototype) of each cluster.
        For interval data, might be shape (k, n_dims, 2).
    distance_func : callable
        A function (sample, center) -> scalar measure. 
        Typically a "distance". If you have "similarity", set isSim=True 
        and inside we do dist=1.0 - sim.
    isSim : bool, optional
        If True, means distance_func is actually returning a similarity. We'll do (1 - sim)
        to interpret it as a distance measure.

    Returns
    -------
    float
        The Davies-Bouldin Index. Smaller => better cluster separation.

    Example
    -------
    # Suppose we used IntervalKMeans => got labels, centers, etc.
    # data shape: (N, n_dims, 2)
    # distance_func: e.g. my_hausdorff
    dbi = Evaluation.davies_bouldin_index(data, labels, kmeans.centroids_, my_hausdorff)
    print("Davies-Bouldin =", dbi)
    """


    if metric in SIMILARITY_FUNCTIONS:
        distance_func = SIMILARITY_FUNCTIONS[metric]
        is_sim = True
    elif metric in DISTANCE_FUNCTIONS:
        distance_func = DISTANCE_FUNCTIONS[metric]
        is_sim = False
    else:
        valid_metric = ", ".join(list(SIMILARITY_FUNCTIONS.keys()) + list(DISTANCE_FUNCTIONS.keys()))
        raise ValueError(f"Invalid metric '{metric}'. Available options: {metric}")

    # 1) cluster_map => which samples in each cluster
    unique_labels = np.unique(labels)
    cluster_map = {}
    for c in unique_labels:
        cluster_map[c] = []
    for i, lab in enumerate(labels):
        cluster_map[lab].append(i)

    k = len(unique_labels)

    # If there's only one cluster or zero => DBI is undefined, return 0 or inf
    if k < 2:
        return 0.0

    # 2) define a small "get_dist" to unify distance vs. similarity
    def dist(x, y):
        val = distance_func(x, y)
        if is_sim:
            return 1.0 - val
        else:
            return val

    # 3) compute S_i = average distance from each sample in cluster i to center i
    # skip empty clusters if any
    S = {}
    for c in unique_labels:
        idxs = cluster_map[c]
        if len(idxs) < 1:
            # skip or treat as 0
            S[c] = 0.0
            continue
        dist_sum = 0.0
        for idx in idxs:
            dist_sum += dist(data[idx], centers[c])
        S[c] = dist_sum / len(idxs)

    # 4) compute D_{i,j} = distance(center_i, center_j)
    # store in a matrix
    kmax = max(unique_labels)+1  # might not be consecutive from 0..k-1, handle carefully
    center_dist = {}
    for i in unique_labels:
        for j in unique_labels:
            if i == j:
                continue
            d_ij = dist(centers[i], centers[j])
            center_dist[(i,j)] = d_ij

    # 5) compute R_{i,j} = (S_i + S_j)/ D_{i,j}
    # then DB_i = max_j( R_{i,j} ) for j != i
    # DB = 1/k sum(DB_i)
    DB_sum = 0.0
    valid_clusters_count = 0
    for i in unique_labels:
        if len(cluster_map[i]) == 0:
            continue
        # skip empty cluster
        max_ij = -1e15
        for j in unique_labels:
            if j == i or len(cluster_map[j])==0:
                continue
            d_ij = center_dist.get((i,j), None)
            if (d_ij is None) or (abs(d_ij) < 1e-15):
                # no distance => skip or treat as large
                continue
            R_ij = (S[i] + S[j]) / d_ij
            if R_ij>max_ij:
                max_ij = R_ij
        if max_ij<0:
            # means no valid j => cluster i alone => DB_i=0 or skip
            continue
        DB_sum += max_ij
        valid_clusters_count += 1

    if valid_clusters_count < 1:
        return 0.0
    dbi = DB_sum / valid_clusters_count
    return dbi
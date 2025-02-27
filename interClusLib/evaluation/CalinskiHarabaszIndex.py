import numpy as np
from collections import defaultdict
from interClusLib.metric import MULTI_SIMILARITY_FUNCTIONS, MULTI_DISTANCE_FUNCTIONS

def calinski_harabasz_index(
        data: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        metric,
    ) -> float:
    """
    Compute the Calinski-Harabasz (CH) Index for a given clustering.

    Parameters
    ----------
    data : np.ndarray
        Shape (N, ...). For interval data => (N, n_dims,2).
    labels : np.ndarray
        Shape (N,). The cluster label for each sample.
    centers : np.ndarray
        Shape (k, ...). The cluster centers (for numeric data or intervals).
        E.g. for interval data => shape(k, n_dims,2).
    distance_func : callable
        A function (sample, center) -> float. 
        If your distance_func is actually similarity, set isSim=True 
        so we do dist=1-sim internally.
    isSim : bool, optional
        If True => interpret distance_func's output as similarity => we do dist=1-sim.
        By default = False => distance_func returns actual distance.

    Returns
    -------
    float
        The Calinski-Harabasz index. Larger => better cluster separation.
        If only one cluster or other degenerate cases => returns 0.

    Notes
    -----
    CH = (B / (k - 1)) / (W / (n - k)),
        where B is "between-cluster scatter", W is "within-cluster scatter".
    For data shape => (N, d,2) in interval scenario, you must define how
    to compute your "distance" and "global centroid". Here we do:
        B = sum_c( n_c * dist^2(centroid_c, centroid_global) )
        W = sum_c sum_{x in c}( dist^2(x, centroid_c) )

    'centroid_global' = average of all data => shape(d,2). 
    'centroid_c' is from 'centers[c]'.

    If using "similarity", we do dist=1-sim => dist^2 for the scatter terms.
    """
    # 1) Check shapes
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length.")

    if metric in MULTI_SIMILARITY_FUNCTIONS:
        distance_func = MULTI_SIMILARITY_FUNCTIONS[metric]
        is_sim = True
    elif metric in MULTI_DISTANCE_FUNCTIONS:
        distance_func = MULTI_DISTANCE_FUNCTIONS[metric]
        is_sim = False
    else:
        valid_metric = ", ".join(list(MULTI_SIMILARITY_FUNCTIONS.keys()) + list(MULTI_DISTANCE_FUNCTIONS.keys()))
        raise ValueError(f"Invalid metric '{metric}'. Available options: {metric}")

    # 2) Basic definitions
    n_samples = len(data)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    if k < 2 or n_samples == k:
        # CH index not well-defined if only 1 cluster or each sample in own cluster
        return 0.0

    # unify function to get distance
    def dist(a, b):
        val = distance_func(a, b)
        if isSim:
            return 1.0 - val
        else:
            return val

    # 3) cluster_map => sample indices in each cluster
    cluster_map = defaultdict(list)
    for i, lab in enumerate(labels):
        cluster_map[lab].append(i)

    # 4) compute the global centroid
    #   in numeric data => global_center = mean(data, axis=0)
    #   in interval data => shape(d,2) => we do dimension-wise mean of lower, mean of upper
    #   => easiest: global_center = average( data along axis=0 )
    global_center = np.mean(data, axis=0)  # shape (...), e.g. (n_dims,2)

    # 5) Compute B: sum_{c} [ n_c * dist^2(centroid_c, global_center) ]
    B = 0.0
    for c in unique_labels:
        n_c = len(cluster_map[c])
        if n_c == 0:
            continue
        dist_c = dist(centers[c], global_center)
        B += n_c * (dist_c**2)

    # 6) Compute W: sum_{c} sum_{ x in c } dist^2( x, centroid_c )
    W = 0.0
    for c in unique_labels:
        centroid_c = centers[c]
        idxs = cluster_map[c]
        for idx in idxs:
            dist_ic = dist(data[idx], centroid_c)
            W += dist_ic**2

    # 7) CH = (B / (k-1)) / (W / (n-k))
    # handle if W=0
    if abs(W)<1e-15:
        # 说明簇内距离全部=0 => all points in each cluster coincide?
        # => CH应该非常大 => return np.inf or 0?
        return 0.0 if B==0 else float('inf')

    CH = (B/(k-1)) / (W/(n_samples - k))
    return CH
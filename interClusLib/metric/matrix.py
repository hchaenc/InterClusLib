import numpy as np
from .similarity import SIMILARITY_FUNCTIONS
from .distance import DISTANCE_FUNCTIONS

def pairwise_similarity(intervals, metric="jaccard"):
    """
    Computes a symmetric (n_samples, n_samples) similarity matrix using the specified metric.

    Parameters:
        intervals: array-like of shape (n_samples, n_dims, 2)
        metric: similarity metric to use ("jaccard", "dice", etc.).
        
    Returns:
        np.ndarray of shape (n_samples, n_samples): similarity matrix.
    """
    # 1. Validate metric
    if metric not in SIMILARITY_FUNCTIONS:
        valid_metrics = ", ".join(SIMILARITY_FUNCTIONS.keys())
        raise ValueError(f"Unsupported metric: {metric}. Available options: {valid_metrics}")

    # 2. Standardize input
    similarity_func = SIMILARITY_FUNCTIONS[metric]
    intervals = np.asarray(intervals, dtype=float)

    # 3. Prepare output matrix
    n = intervals.shape[0]
    sim_matrix = np.zeros((n, n), dtype=float)

    # 4. Compute only upper triangle + diagonal
    for i in range(n):
        sim_matrix[i, i] = 1.0  # self‐similarity
        for j in range(i + 1, n):
            sim = similarity_func(intervals[i], intervals[j])
            sim_matrix[i, j] = sim_matrix[j, i] = sim

    return sim_matrix

def pairwise_distance(intervals, metric="hausdorff"):
    """
    Computes a symmetric (n_samples, n_samples) distance matrix using the specified metric.

    :param intervals: array of shape (n_samples, n_dims, 2)
    :param metric: distance metric to use ("hausdorff", "euclidean", "manhattan").
    :return: (n_samples, n_samples) distance matrix.
    """
    if metric not in DISTANCE_FUNCTIONS:
        valid_metrics = ", ".join(DISTANCE_FUNCTIONS.keys())
        raise ValueError(f"Unsupported metric: {metric}. Available options: {valid_metrics}")

    distance_func = DISTANCE_FUNCTIONS[metric]
    intervals = np.asarray(intervals)
    n_samples = intervals.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i, n_samples):  # Only compute upper triangle
            if i == j:
                dist = 0.0
            else:
                dist = distance_func(intervals[i], intervals[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # Mirror for symmetry

    return dist_matrix

def cross_similarity(intervals_a, intervals_b, metric="jaccard"):
    """
    Computes a (M, N) cross-similarity matrix between two sets of interval data.

    Parameters:
        intervals_a: array-like of shape (M, n_dims, 2)
            First set of intervals.
        intervals_b: array-like of shape (N, n_dims, 2)
            Second set of intervals.
        metric: str, default="jaccard"
            Similarity metric to use ("jaccard", "dice", etc.).

    Returns:
        np.ndarray of shape (M, N): cross-similarity matrix.
    """
    # 1. Validate metric choice
    if metric not in SIMILARITY_FUNCTIONS:
        valid = ", ".join(SIMILARITY_FUNCTIONS.keys())
        raise ValueError(f"Unsupported metric: {metric}. Available options: {valid}")

    # 2. Fetch the similarity function and standardize inputs
    sim_func = SIMILARITY_FUNCTIONS[metric]
    A = np.asarray(intervals_a, dtype=float)
    B = np.asarray(intervals_b, dtype=float)

    # 3. Prepare output matrix
    M, N = A.shape[0], B.shape[0]
    sim_matrix = np.zeros((M, N), dtype=float)

    # 4. Compute cross-similarity
    for i in range(M):
        row_i = A[i]
        for j in range(N):
            sim_matrix[i, j] = sim_func(row_i, B[j])

    return sim_matrix

def cross_distance(intervals_a, intervals_b, metric="hausdorff"):
    """
    Computes a (M, N) cross-distance matrix between two sets of interval data.

    :param intervals_a: array of shape (M, n_dims, 2) - First set of intervals.
    :param intervals_b: array of shape (N, n_dims, 2) - Second set of intervals.
    :param metric: distance metric to use ("hausdorff", "euclidean", "manhattan").
    :return: (M, N) distance matrix.
    """
    if metric not in DISTANCE_FUNCTIONS:
        valid_metrics = ", ".join(DISTANCE_FUNCTIONS.keys())
        raise ValueError(f"Unsupported metric: {metric}. Available options: {valid_metrics}")

    distance_func = DISTANCE_FUNCTIONS[metric]
    intervals_a = np.asarray(intervals_a)
    intervals_b = np.asarray(intervals_b)

    m, n = intervals_a.shape[0], intervals_b.shape[0]
    dist_matrix = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            dist_matrix[i, j] = distance_func(intervals_a[i], intervals_b[j])

    return dist_matrix
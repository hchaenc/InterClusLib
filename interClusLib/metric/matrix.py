import numpy as np
from .similarity import SIMILARITY_FUNCTIONS
from .distance import DISTANCE_FUNCTIONS

def pairwise_similarity(intervals, metric="jaccard"):
    """
    Computes an (n_samples, n_samples) similarity matrix.

    :param intervals: array of shape (n_samples, n_dims, 2)
    :param metric: similarity metric to use ("jaccard", "dice", etc.).
    :return: (n_samples, n_samples) similarity matrix.
    """
    if metric not in SIMILARITY_FUNCTIONS:
        valid_metrics = ", ".join(SIMILARITY_FUNCTIONS.keys())
        raise ValueError(f"Unsupported metric: {metric}. Available options: {valid_metrics}")

    similarity_func = SIMILARITY_FUNCTIONS[metric]
    n_samples = intervals.shape[0]
    sim_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            sim_matrix[i, j] = similarity_func(intervals[i], intervals[j])

    return sim_matrix

def pairwise_distance(intervals, metric="hausdorff"):
    """
    Computes an (n_samples, n_samples) distance matrix.

    :param intervals: array of shape (n_samples, n_dims, 2)
    :param metric: distance metric to use ("hausdorff", "euclidean", "manhattan").
    :return: (n_samples, n_samples) distance matrix.
    """
    if metric not in DISTANCE_FUNCTIONS:
        valid_metrics = ", ".join(DISTANCE_FUNCTIONS.keys())
        raise ValueError(f"Unsupported metric: {metric}. Available options: {valid_metrics}")

    distance_func = DISTANCE_FUNCTIONS[metric]
    n_samples = intervals.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            dist_matrix[i, j] = distance_func(intervals[i], intervals[j])

    return dist_matrix

def cross_similarity(intervals_a, intervals_b, metric="jaccard"):
    """
    Computes a (M, N) cross-similarity matrix between two sets of interval data.

    :param intervals_a: array of shape (M, n_dims, 2) - First set of intervals.
    :param intervals_b: array of shape (N, n_dims, 2) - Second set of intervals.
    :param metric: similarity metric to use ("jaccard", "dice", etc.).
    :return: (M, N) similarity matrix.
    """
    if metric not in SIMILARITY_FUNCTIONS:
        valid_metrics = ", ".join(SIMILARITY_FUNCTIONS.keys())
        raise ValueError(f"Unsupported metric: {metric}. Available options: {valid_metrics}")

    similarity_func = SIMILARITY_FUNCTIONS[metric]
    m_samples = intervals_a.shape[0]
    n_samples = intervals_b.shape[0]
    sim_matrix = np.zeros((m_samples, n_samples))

    for i in range(m_samples):
        for j in range(n_samples):
            sim_matrix[i, j] = similarity_func(intervals_a[i], intervals_b[j])

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
    m_samples = intervals_a.shape[0]
    n_samples = intervals_b.shape[0]
    dist_matrix = np.zeros((m_samples, n_samples))

    for i in range(m_samples):
        for j in range(n_samples):
            dist_matrix[i, j] = distance_func(intervals_a[i], intervals_b[j])

    return dist_matrix
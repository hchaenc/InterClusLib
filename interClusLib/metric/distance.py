import numpy as np

# ========== 2. Single-Dimension Interval Distance Measures ==========
def hausdorff_distance(interval1, interval2):
    """
    Calculate the Hausdorff distance between two interval values.

    Parameters:
        interval1 (np.array): Interval value 1.
        interval2 (np.array): Interval value 2.

    Returns:
        float: Hausdorff distance between the two interval values.
    """
    # calculate the Hausdorff distance between two interval values
    diff2 = interval2[1] - interval1[1]
    diff1 = interval2[0] - interval1[0]

    return max(abs(diff2), abs(diff1))   

def euclidean_distance(interval1, interval2):
    """
    Calculate the Euclidean distance between the range of two interval values.

    Parameters:
        interval1 (np.array): Interval value 1.
        interval2 (np.array): Interval value 2.

    Returns:
        float: Euclidean distance between the range of the two interval values.
    """
    # calculate the Euclidean distance between the range of two interval values
    diff1 = interval1[0] - interval2[0]
    diff2 = interval1[1] - interval2[1]

    return np.sqrt(diff1**2 + diff2**2)

def manhattan_distance(interval1, interval2):
    """
    Calculate the Manhattan distance between two interval values.

    Parameters:
        interval1 (np.array): Interval value 1.
        interval2 (np.array): Interval value 2.

    Returns:
        float: Manhattan distance between the two interval values.
    """
    # calculate the Manhattan distance between two interval values
    diff2 = interval2[1] - interval1[1]
    diff1 = interval2[0] - interval1[0]

    return abs(diff1) + abs(diff2)

def hausdorff_distance_md(interval_a, interval_b):
    """
    Multi-dimensional Hausdorff distance.

    :param interval_a: shape (n_dims, 2)
    :param interval_b: shape (n_dims, 2)
    :return: scalar distance
    """
    n_dims = interval_a.shape[0]
    distances = []
    for d in range(n_dims):
        dist_1d = hausdorff_distance(interval_a[d], interval_b[d])
        distances.append(dist_1d)
    
    return np.sum(distances)

def euclidean_distance_md(interval_a, interval_b):
    """
    Multi-dimensional range euclidean distance.

    :param interval_a: shape (n_dims, 2)
    :param interval_b: shape (n_dims, 2)
    :return: scalar distance
    """
    n_dims = interval_a.shape[0]
    distances = []
    for d in range(n_dims):
        dist_1d = euclidean_distance(interval_a[d], interval_b[d])
        distances.append(dist_1d * dist_1d)

    return np.sqrt(np.sum(distances))

def manhattan_distance_md(interval_a, interval_b):
    """
    Multi-dimensional manhattan distance.

    :param interval_a: shape (n_dims, 2)
    :param interval_b: shape (n_dims, 2)
    :return: scalar distance
    """
    n_dims = interval_a.shape[0]
    distances = []
    for d in range(n_dims):
        dist_1d = manhattan_distance(interval_a[d], interval_b[d])
        distances.append(dist_1d)

    return np.sum(distances)

DISTANCE_FUNCTIONS = {
    "hausdorff": hausdorff_distance,
    "manhattan": manhattan_distance,
    "euclidean": euclidean_distance,
}

MULTI_DISTANCE_FUNCTIONS = {
    "hausdorff": hausdorff_distance_md,
    "manhattan": manhattan_distance_md,
    "euclidean": euclidean_distance_md,
}
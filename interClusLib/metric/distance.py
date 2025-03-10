import numpy as np

def hausdorff_distance(interval1, interval2):
    """
    Calculate the Hausdorff distance between two intervals.
    
    Parameters:
        interval1: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        interval2: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        
    Returns:
        float: Hausdorff distance
    """
    # Check if inputs are multi-dimensional
    if len(interval1.shape) > 1:
        n_dims = interval1.shape[0]
        distances = []
        for d in range(n_dims):
            diff1 = abs(interval2[d, 0] - interval1[d, 0])
            diff2 = abs(interval2[d, 1] - interval1[d, 1])
            distances.append(max(diff1, diff2))
        return np.sum(distances)
    else:
        # Single-dimensional case
        diff1 = abs(interval2[0] - interval1[0])
        diff2 = abs(interval2[1] - interval1[1])
        return max(diff1, diff2)

def euclidean_distance(interval1, interval2):
    """
    Calculate the Euclidean distance between two intervals.
    
    Parameters:
        interval1: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        interval2: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        
    Returns:
        float: Euclidean distance
    """
    # Check if inputs are multi-dimensional
    if len(interval1.shape) > 1:
        n_dims = interval1.shape[0]
        squared_distances = []
        for d in range(n_dims):
            diff1 = interval1[d, 0] - interval2[d, 0]
            diff2 = interval1[d, 1] - interval2[d, 1]
            squared_distances.append(diff1**2 + diff2**2)
        return np.sqrt(np.sum(squared_distances))
    else:
        # Single-dimensional case
        diff1 = interval1[0] - interval2[0]
        diff2 = interval1[1] - interval2[1]
        return np.sqrt(diff1**2 + diff2**2)

def manhattan_distance(interval1, interval2):
    """
    Calculate the Manhattan distance between two intervals.
    
    Parameters:
        interval1: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        interval2: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        
    Returns:
        float: Manhattan distance
    """
    # Check if inputs are multi-dimensional
    if len(interval1.shape) > 1:
        n_dims = interval1.shape[0]
        distances = []
        for d in range(n_dims):
            diff1 = abs(interval2[d, 0] - interval1[d, 0])
            diff2 = abs(interval2[d, 1] - interval1[d, 1])
            distances.append(diff1 + diff2)
        return np.sum(distances)
    else:
        # Single-dimensional case
        diff1 = abs(interval2[0] - interval1[0])
        diff2 = abs(interval2[1] - interval1[1])
        return diff1 + diff2

# Dictionary mapping distance type to distance function
DISTANCE_FUNCTIONS = {
    "hausdorff": hausdorff_distance,
    "manhattan": manhattan_distance,
    "euclidean": euclidean_distance,
}
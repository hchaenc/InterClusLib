import numpy as np

def hausdorff_distance(interval1, interval2):
    """
    Calculate the Hausdorff distance between two intervals.
    
    Parameters:
        interval1: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples1, n_dims, 2)
        interval2: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples2, n_dims, 2)
        
    Returns:
        float or np.ndarray: Hausdorff distance(s)
                           - scalar if both inputs are single intervals
                           - vector if one input is batch
                           - matrix if both inputs are batches
    """
    # Convert inputs to numpy arrays
    interval1 = np.asarray(interval1, dtype=np.float64)
    interval2 = np.asarray(interval2, dtype=np.float64)
    
    # Handle different input dimensions
    single1 = interval1.ndim <= 2
    single2 = interval2.ndim <= 2
    
    # For legacy compatibility with original function
    if interval1.ndim == 1 or interval2.ndim == 1:
        # Single 1D interval case
        if interval1.ndim == 1 and interval2.ndim == 1:
            return np.max(np.abs(interval2 - interval1))
        # Mixed 1D and multi-dimensional case
        if interval1.ndim == 1:
            interval1 = interval1[None, :]
        if interval2.ndim == 1:
            interval2 = interval2[None, :]
        diffs = np.abs(interval2 - interval1)
        max_diffs = np.max(diffs, axis=1)
        return np.sum(max_diffs)
    
    # Standardize to 3D format: (n_samples, n_dims, 2)
    if interval1.ndim == 2:
        interval1 = interval1[None, :, :]
    if interval2.ndim == 2:
        interval2 = interval2[None, :, :]
    
    # Vectorized computation using broadcasting
    # Shape: (n_samples1, 1, n_dims, 2) - (1, n_samples2, n_dims, 2) = (n_samples1, n_samples2, n_dims, 2)
    abs_diff = np.abs(interval1[:, None, :, :] - interval2[None, :, :, :])
    
    # For each dimension, take the maximum difference between interval bounds
    max_diff_per_dim = np.max(abs_diff, axis=3)  # Shape: (n_samples1, n_samples2, n_dims)
    
    # Sum across all dimensions
    distances = np.sum(max_diff_per_dim, axis=2)  # Shape: (n_samples1, n_samples2)
    
    # Format output based on input types
    if single1 and single2:
        return distances[0, 0]
    elif single1:
        return distances[0, :]
    elif single2:
        return distances[:, 0]
    else:
        return distances

def euclidean_distance(interval1, interval2):
    """
    Calculate the Euclidean distance between two intervals.
    
    Parameters:
        interval1: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples1, n_dims, 2)
        interval2: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples2, n_dims, 2)
        
    Returns:
        float or np.ndarray: Euclidean distance(s)
                           - scalar if both inputs are single intervals
                           - vector if one input is batch
                           - matrix if both inputs are batches
    """
    # Convert inputs to numpy arrays
    interval1 = np.asarray(interval1, dtype=np.float64)
    interval2 = np.asarray(interval2, dtype=np.float64)
    
    # Handle different input dimensions
    single1 = interval1.ndim <= 2
    single2 = interval2.ndim <= 2
    
    # For legacy compatibility with original function
    if interval1.ndim <= 2 and interval2.ndim <= 2:
        diffs = interval1 - interval2
        return np.sqrt(np.sum(diffs ** 2))
    
    # Standardize to 3D format: (n_samples, n_dims, 2)
    if interval1.ndim == 2:
        interval1 = interval1[None, :, :]
    if interval2.ndim == 2:
        interval2 = interval2[None, :, :]
    
    # Vectorized computation using broadcasting
    # Method 1: Direct computation (memory intensive but simple)
    diff = interval1[:, None, :, :] - interval2[None, :, :, :]
    distances_squared = np.sum(diff**2, axis=(2, 3))
    distances = np.sqrt(distances_squared + 1e-16)  # Add epsilon for numerical stability
    
    # Format output based on input types
    if single1 and single2:
        return distances[0, 0]
    elif single1:
        return distances[0, :]
    elif single2:
        return distances[:, 0]
    else:
        return distances

def euclidean_distance_optimized(interval1, interval2):
    """
    Calculate the Euclidean distance between two intervals using optimized computation.
    Uses the mathematical identity: ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
    More memory efficient for large datasets.
    
    Parameters:
        interval1: Batch of intervals with shape (n_samples1, n_dims, 2)
        interval2: Batch of intervals with shape (n_samples2, n_dims, 2)
        
    Returns:
        np.ndarray: Euclidean distance matrix of shape (n_samples1, n_samples2)
    """
    # Convert inputs to numpy arrays
    interval1 = np.asarray(interval1, dtype=np.float64)
    interval2 = np.asarray(interval2, dtype=np.float64)
    
    # Standardize to 3D format
    single1 = interval1.ndim == 2
    single2 = interval2.ndim == 2
    
    if single1:
        interval1 = interval1[None, :, :]
    if single2:
        interval2 = interval2[None, :, :]
    
    # Compute squared norms: ||x||²
    norm1_squared = np.einsum('ijk,ijk->i', interval1, interval1)  # Shape: (n_samples1,)
    norm2_squared = np.einsum('ijk,ijk->i', interval2, interval2)  # Shape: (n_samples2,)
    
    # Compute cross terms: 2⟨x,y⟩
    cross_term = 2 * np.einsum('ijk,ljk->il', interval1, interval2)  # Shape: (n_samples1, n_samples2)
    
    # Compute squared distances: ||x||² + ||y||² - 2⟨x,y⟩
    distances_squared = norm1_squared[:, None] + norm2_squared[None, :] - cross_term
    distances_squared = np.maximum(distances_squared, 0)  # Ensure non-negative due to numerical errors
    
    # Take square root
    distances = np.sqrt(distances_squared + 1e-16)
    
    # Format output based on input types
    if single1 and single2:
        return distances[0, 0]
    elif single1:
        return distances[0, :]
    elif single2:
        return distances[:, 0]
    else:
        return distances

def manhattan_distance(interval1, interval2):
    """
    Calculate the Manhattan distance between two intervals.
    
    Parameters:
        interval1: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples1, n_dims, 2)
        interval2: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples2, n_dims, 2)
        
    Returns:
        float or np.ndarray: Manhattan distance(s)
                           - scalar if both inputs are single intervals
                           - vector if one input is batch
                           - matrix if both inputs are batches
    """
    # Convert inputs to numpy arrays
    interval1 = np.asarray(interval1, dtype=np.float64)
    interval2 = np.asarray(interval2, dtype=np.float64)
    
    # Handle different input dimensions
    single1 = interval1.ndim <= 2
    single2 = interval2.ndim <= 2
    
    # For legacy compatibility with original function
    if interval1.ndim <= 2 and interval2.ndim <= 2:
        diffs = np.abs(interval1 - interval2)
        return np.sum(diffs)
    
    # Standardize to 3D format: (n_samples, n_dims, 2)
    if interval1.ndim == 2:
        interval1 = interval1[None, :, :]
    if interval2.ndim == 2:
        interval2 = interval2[None, :, :]
    
    # Vectorized computation using broadcasting
    abs_diff = np.abs(interval1[:, None, :, :] - interval2[None, :, :, :])
    distances = np.sum(abs_diff, axis=(2, 3))
    
    # Format output based on input types
    if single1 and single2:
        return distances[0, 0]
    elif single1:
        return distances[0, :]
    elif single2:
        return distances[:, 0]
    else:
        return distances

def chebyshev_distance(interval1, interval2):
    """
    Calculate the Chebyshev distance (L∞ norm) between two intervals.
    
    Parameters:
        interval1: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples1, n_dims, 2)
        interval2: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples2, n_dims, 2)
        
    Returns:
        float or np.ndarray: Chebyshev distance(s)
                           - scalar if both inputs are single intervals
                           - vector if one input is batch
                           - matrix if both inputs are batches
    """
    # Convert inputs to numpy arrays
    interval1 = np.asarray(interval1, dtype=np.float64)
    interval2 = np.asarray(interval2, dtype=np.float64)
    
    # Handle different input dimensions
    single1 = interval1.ndim <= 2
    single2 = interval2.ndim <= 2
    
    # For simple cases
    if interval1.ndim <= 2 and interval2.ndim <= 2:
        abs_diff = np.abs(interval1 - interval2)
        return np.max(abs_diff)
    
    # Standardize to 3D format: (n_samples, n_dims, 2)
    if interval1.ndim == 2:
        interval1 = interval1[None, :, :]
    if interval2.ndim == 2:
        interval2 = interval2[None, :, :]
    
    # Vectorized computation using broadcasting
    abs_diff = np.abs(interval1[:, None, :, :] - interval2[None, :, :, :])
    distances = np.max(abs_diff, axis=(2, 3))
    
    # Format output based on input types
    if single1 and single2:
        return distances[0, 0]
    elif single1:
        return distances[0, :]
    elif single2:
        return distances[:, 0]
    else:
        return distances

def cosine_distance(interval1, interval2):
    """
    Calculate the cosine distance between two intervals.
    Cosine distance = 1 - cosine_similarity
    
    Parameters:
        interval1: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples1, n_dims, 2)
        interval2: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples2, n_dims, 2)
        
    Returns:
        float or np.ndarray: Cosine distance(s) in range [0, 2]
                           - scalar if both inputs are single intervals
                           - vector if one input is batch
                           - matrix if both inputs are batches
    """
    # Convert inputs to numpy arrays
    interval1 = np.asarray(interval1, dtype=np.float64)
    interval2 = np.asarray(interval2, dtype=np.float64)
    
    # Handle different input dimensions
    single1 = interval1.ndim <= 2
    single2 = interval2.ndim <= 2
    
    # For simple cases, flatten intervals to vectors
    if interval1.ndim <= 2 and interval2.ndim <= 2:
        flat1 = interval1.flatten()
        flat2 = interval2.flatten()
        
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance for zero vectors
        
        cosine_sim = np.dot(flat1, flat2) / (norm1 * norm2)
        return 1.0 - cosine_sim
    
    # Standardize to 3D format: (n_samples, n_dims, 2)
    if interval1.ndim == 2:
        interval1 = interval1[None, :, :]
    if interval2.ndim == 2:
        interval2 = interval2[None, :, :]
    
    # Flatten intervals to vectors for cosine computation
    flat1 = interval1.reshape(interval1.shape[0], -1)  # Shape: (n_samples1, n_dims*2)
    flat2 = interval2.reshape(interval2.shape[0], -1)  # Shape: (n_samples2, n_dims*2)
    
    # Compute norms
    norm1 = np.linalg.norm(flat1, axis=1, keepdims=True)  # Shape: (n_samples1, 1)
    norm2 = np.linalg.norm(flat2, axis=1, keepdims=True)  # Shape: (n_samples2, 1)
    
    # Compute dot products
    dot_products = np.dot(flat1, flat2.T)  # Shape: (n_samples1, n_samples2)
    
    # Compute cosine similarities
    cosine_similarities = dot_products / (norm1 * norm2.T + 1e-16)
    
    # Compute cosine distances
    distances = 1.0 - cosine_similarities
    distances = np.clip(distances, 0, 2)  # Ensure valid range [0, 2]
    
    # Format output based on input types
    if single1 and single2:
        return distances[0, 0]
    elif single1:
        return distances[0, :]
    elif single2:
        return distances[:, 0]
    else:
        return distances

# Dictionary mapping distance type to distance function
DISTANCE_FUNCTIONS = {
    "hausdorff": hausdorff_distance,
    "manhattan": manhattan_distance,
    "euclidean": euclidean_distance,
    "euclidean_optimized": euclidean_distance_optimized,
    "chebyshev": chebyshev_distance,
    "cosine": cosine_distance,
}
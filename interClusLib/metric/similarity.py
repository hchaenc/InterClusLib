import numpy as np

def jaccard_similarity(interval1, interval2):
    """
    Calculate the Jaccard similarity between two intervals (1D or multi-D).

    Parameters:
        interval1: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples1, n_dims, 2)
        interval2: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples2, n_dims, 2)

    Returns:
        float or np.ndarray: Jaccard similarity/similarities
                           - scalar if both inputs are single intervals
                           - vector if one input is batch
                           - matrix if both inputs are batches
    """
    # Convert inputs to numpy arrays
    a = np.asarray(interval1, dtype=np.float64)
    b = np.asarray(interval2, dtype=np.float64)
    
    # Handle different input dimensions
    single1 = a.ndim <= 2
    single2 = b.ndim <= 2
    
    # Legacy compatibility for simple cases
    if a.ndim <= 2 and b.ndim <= 2:
        # Ensure the shape is (n_dims, 2); promote 1D to 2D if necessary
        if a.ndim == 1:
            a = a[np.newaxis, :]
            b = b[np.newaxis, :]
        
        # Split into lower and upper bounds
        lower_a, upper_a = a[:, 0], a[:, 1]
        lower_b, upper_b = b[:, 0], b[:, 1]
        
        # Compute intersection length for each dimension
        intersection = np.maximum(0, np.minimum(upper_a, upper_b) - np.maximum(lower_a, lower_b))
        
        # Compute union length for each dimension
        length_a = upper_a - lower_a
        length_b = upper_b - lower_b
        union = length_a + length_b - intersection
        
        # Compute Jaccard similarity per dimension; set to 0 when union is 0
        sim = np.where(union > 0, intersection / union, 0.0)
        
        # Return the average similarity across dimensions
        return sim.mean()
    
    # Standardize to 3D format: (n_samples, n_dims, 2)
    if a.ndim == 2:
        a = a[None, :, :]
    if b.ndim == 2:
        b = b[None, :, :]
    
    # Vectorized computation using broadcasting
    # Extract bounds: (n_samples1, 1, n_dims) and (1, n_samples2, n_dims)
    lower_a, upper_a = a[:, None, :, 0], a[:, None, :, 1]
    lower_b, upper_b = b[None, :, :, 0], b[None, :, :, 1]
    
    # Compute intersection bounds: (n_samples1, n_samples2, n_dims)
    intersection_lower = np.maximum(lower_a, lower_b)
    intersection_upper = np.minimum(upper_a, upper_b)
    intersection = np.maximum(0, intersection_upper - intersection_lower)
    
    # Compute union lengths: (n_samples1, n_samples2, n_dims)
    length_a = upper_a - lower_a
    length_b = upper_b - lower_b
    union = length_a + length_b - intersection
    
    # Compute Jaccard similarity per dimension
    sim_per_dim = np.where(union > 0, intersection / union, 0.0)
    
    # Average across dimensions: (n_samples1, n_samples2)
    similarities = np.mean(sim_per_dim, axis=2)
    
    # Format output based on input types
    if single1 and single2:
        return similarities[0, 0]
    elif single1:
        return similarities[0, :]
    elif single2:
        return similarities[:, 0]
    else:
        return similarities

def dice_similarity(interval1, interval2):
    """
    Calculate the Dice similarity between two intervals (1D or multi-D).

    Parameters:
        interval1: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples1, n_dims, 2)
        interval2: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples2, n_dims, 2)

    Returns:
        float or np.ndarray: Dice similarity/similarities
                           - scalar if both inputs are single intervals
                           - vector if one input is batch
                           - matrix if both inputs are batches
    """
    # Convert inputs to numpy arrays
    a = np.asarray(interval1, dtype=np.float64)
    b = np.asarray(interval2, dtype=np.float64)
    
    # Handle different input dimensions
    single1 = a.ndim <= 2
    single2 = b.ndim <= 2
    
    # Legacy compatibility for simple cases
    if a.ndim <= 2 and b.ndim <= 2:
        # Ensure the shape is (n_dims, 2); promote 1D intervals to 2D
        if a.ndim == 1:
            a = a[np.newaxis, :]
            b = b[np.newaxis, :]
        
        # Split into lower and upper bounds
        lower_a, upper_a = a[:, 0], a[:, 1]
        lower_b, upper_b = b[:, 0], b[:, 1]
        
        # Compute intersection length for each dimension
        intersection = np.maximum(0, np.minimum(upper_a, upper_b) - np.maximum(lower_a, lower_b))
        
        # Compute sum of interval lengths for each dimension
        length_a = upper_a - lower_a
        length_b = upper_b - lower_b
        sum_lengths = length_a + length_b
        
        # Compute Dice similarity per dimension; set to 0 where sum_lengths is 0
        sim = np.where(sum_lengths > 0, 2 * intersection / sum_lengths, 0.0)
        
        # Return the average similarity across dimensions
        return sim.mean()
    
    # Standardize to 3D format: (n_samples, n_dims, 2)
    if a.ndim == 2:
        a = a[None, :, :]
    if b.ndim == 2:
        b = b[None, :, :]
    
    # Vectorized computation using broadcasting
    # Extract bounds: (n_samples1, 1, n_dims) and (1, n_samples2, n_dims)
    lower_a, upper_a = a[:, None, :, 0], a[:, None, :, 1]
    lower_b, upper_b = b[None, :, :, 0], b[None, :, :, 1]
    
    # Compute intersection bounds: (n_samples1, n_samples2, n_dims)
    intersection_lower = np.maximum(lower_a, lower_b)
    intersection_upper = np.minimum(upper_a, upper_b)
    intersection = np.maximum(0, intersection_upper - intersection_lower)
    
    # Compute sum of lengths: (n_samples1, n_samples2, n_dims)
    length_a = upper_a - lower_a
    length_b = upper_b - lower_b
    sum_lengths = length_a + length_b
    
    # Compute Dice similarity per dimension
    sim_per_dim = np.where(sum_lengths > 0, 2 * intersection / sum_lengths, 0.0)
    
    # Average across dimensions: (n_samples1, n_samples2)
    similarities = np.mean(sim_per_dim, axis=2)
    
    # Format output based on input types
    if single1 and single2:
        return similarities[0, 0]
    elif single1:
        return similarities[0, :]
    elif single2:
        return similarities[:, 0]
    else:
        return similarities

def bidirectional_similarity_min(interval1, interval2):
    """
    Calculate the minimum bidirectional subset similarity between two intervals (1D or multi-D).

    Parameters:
        interval1: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples1, n_dims, 2)
        interval2: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples2, n_dims, 2)

    Returns:
        float or np.ndarray: Minimum bidirectional subset similarity/similarities
                           - scalar if both inputs are single intervals
                           - vector if one input is batch
                           - matrix if both inputs are batches
    """
    # Convert inputs to numpy arrays
    a = np.asarray(interval1, dtype=np.float64)
    b = np.asarray(interval2, dtype=np.float64)
    
    # Handle different input dimensions
    single1 = a.ndim <= 2
    single2 = b.ndim <= 2
    
    # Legacy compatibility for simple cases
    if a.ndim <= 2 and b.ndim <= 2:
        # Promote 1D intervals to 2D shape (n_dims, 2)
        if a.ndim == 1:
            a = a[np.newaxis, :]
            b = b[np.newaxis, :]
        
        # Split into lower and upper bounds arrays
        lower_a, upper_a = a[:, 0], a[:, 1]
        lower_b, upper_b = b[:, 0], b[:, 1]
        
        # Compute intersection length for each dimension
        intersection = np.maximum(0, np.minimum(upper_a, upper_b) - np.maximum(lower_a, lower_b))
        
        # Compute full lengths of each interval
        length_a = upper_a - lower_a
        length_b = upper_b - lower_b
        
        # Compute non-overlap lengths in both directions
        non_overlap_a_b = np.maximum(0, length_a - intersection)
        non_overlap_b_a = np.maximum(0, length_b - intersection)
        
        # Compute denominators for both directions, avoiding division by zero
        denom_a_b = intersection + non_overlap_a_b
        denom_b_a = intersection + non_overlap_b_a
        
        # Compute directional subset similarities; set to 0 where denominator is zero
        rec_a_b = np.where(denom_a_b > 0, intersection / denom_a_b, 0.0)
        rec_b_a = np.where(denom_b_a > 0, intersection / denom_b_a, 0.0)
        
        # Minimum bidirectional similarity per dimension
        sim = np.minimum(rec_a_b, rec_b_a)
        
        # Return average over all dimensions
        return sim.mean()
    
    # Standardize to 3D format: (n_samples, n_dims, 2)
    if a.ndim == 2:
        a = a[None, :, :]
    if b.ndim == 2:
        b = b[None, :, :]
    
    # Vectorized computation using broadcasting
    # Extract bounds: (n_samples1, 1, n_dims) and (1, n_samples2, n_dims)
    lower_a, upper_a = a[:, None, :, 0], a[:, None, :, 1]
    lower_b, upper_b = b[None, :, :, 0], b[None, :, :, 1]
    
    # Compute intersection bounds: (n_samples1, n_samples2, n_dims)
    intersection_lower = np.maximum(lower_a, lower_b)
    intersection_upper = np.minimum(upper_a, upper_b)
    intersection = np.maximum(0, intersection_upper - intersection_lower)
    
    # Compute full lengths of each interval: (n_samples1, n_samples2, n_dims)
    length_a = upper_a - lower_a
    length_b = upper_b - lower_b
    
    # Compute non-overlap lengths in both directions
    non_overlap_a_b = np.maximum(0, length_a - intersection)
    non_overlap_b_a = np.maximum(0, length_b - intersection)
    
    # Compute denominators for both directions
    denom_a_b = intersection + non_overlap_a_b
    denom_b_a = intersection + non_overlap_b_a
    
    # Compute directional subset similarities
    rec_a_b = np.where(denom_a_b > 0, intersection / denom_a_b, 0.0)
    rec_b_a = np.where(denom_b_a > 0, intersection / denom_b_a, 0.0)
    
    # Minimum bidirectional similarity per dimension
    sim_per_dim = np.minimum(rec_a_b, rec_b_a)
    
    # Average across dimensions: (n_samples1, n_samples2)
    similarities = np.mean(sim_per_dim, axis=2)
    
    # Format output based on input types
    if single1 and single2:
        return similarities[0, 0]
    elif single1:
        return similarities[0, :]
    elif single2:
        return similarities[:, 0]
    else:
        return similarities

def bidirectional_similarity_prod(interval1, interval2):
    """
    Calculate the product bidirectional subset similarity between two intervals (1D or multi-D).

    Parameters:
        interval1: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples1, n_dims, 2)
        interval2: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples2, n_dims, 2)

    Returns:
        float or np.ndarray: Product of bidirectional subset similarities
                           - scalar if both inputs are single intervals
                           - vector if one input is batch
                           - matrix if both inputs are batches
    """
    # Convert inputs to numpy arrays
    a = np.asarray(interval1, dtype=np.float64)
    b = np.asarray(interval2, dtype=np.float64)
    
    # Handle different input dimensions
    single1 = a.ndim <= 2
    single2 = b.ndim <= 2
    
    # Legacy compatibility for simple cases
    if a.ndim <= 2 and b.ndim <= 2:
        # Promote 1D intervals to 2D shape (n_dims, 2)
        if a.ndim == 1:
            a = a[np.newaxis, :]
            b = b[np.newaxis, :]
        
        # Split into lower and upper bounds
        lower_a, upper_a = a[:, 0], a[:, 1]
        lower_b, upper_b = b[:, 0], b[:, 1]
        
        # Compute intersection length for each dimension
        intersection = np.maximum(0, np.minimum(upper_a, upper_b) - np.maximum(lower_a, lower_b))
        
        # Compute full lengths of each interval
        length_a = upper_a - lower_a
        length_b = upper_b - lower_b
        
        # Compute non-overlap lengths in both directions
        non_overlap_a_b = np.maximum(0, length_a - intersection)
        non_overlap_b_a = np.maximum(0, length_b - intersection)
        
        # Compute denominators for directional ratios, avoiding division by zero
        denom_a_b = intersection + non_overlap_a_b
        denom_b_a = intersection + non_overlap_b_a
        
        # Compute directional subset similarities; set to 0 where denominator is zero
        rec_a_b = np.where(denom_a_b > 0, intersection / denom_a_b, 0.0)
        rec_b_a = np.where(denom_b_a > 0, intersection / denom_b_a, 0.0)
        
        # Compute product similarity per dimension
        sim = rec_a_b * rec_b_a
        
        # Return the mean similarity over all dimensions
        return sim.mean()
    
    # Standardize to 3D format: (n_samples, n_dims, 2)
    if a.ndim == 2:
        a = a[None, :, :]
    if b.ndim == 2:
        b = b[None, :, :]
    
    # Vectorized computation using broadcasting
    # Extract bounds: (n_samples1, 1, n_dims) and (1, n_samples2, n_dims)
    lower_a, upper_a = a[:, None, :, 0], a[:, None, :, 1]
    lower_b, upper_b = b[None, :, :, 0], b[None, :, :, 1]
    
    # Compute intersection bounds: (n_samples1, n_samples2, n_dims)
    intersection_lower = np.maximum(lower_a, lower_b)
    intersection_upper = np.minimum(upper_a, upper_b)
    intersection = np.maximum(0, intersection_upper - intersection_lower)
    
    # Compute full lengths of each interval: (n_samples1, n_samples2, n_dims)
    length_a = upper_a - lower_a
    length_b = upper_b - lower_b
    
    # Compute non-overlap lengths in both directions
    non_overlap_a_b = np.maximum(0, length_a - intersection)
    non_overlap_b_a = np.maximum(0, length_b - intersection)
    
    # Compute denominators for directional ratios
    denom_a_b = intersection + non_overlap_a_b
    denom_b_a = intersection + non_overlap_b_a
    
    # Compute directional subset similarities
    rec_a_b = np.where(denom_a_b > 0, intersection / denom_a_b, 0.0)
    rec_b_a = np.where(denom_b_a > 0, intersection / denom_b_a, 0.0)
    
    # Compute product similarity per dimension
    sim_per_dim = rec_a_b * rec_b_a
    
    # Average across dimensions: (n_samples1, n_samples2)
    similarities = np.mean(sim_per_dim, axis=2)
    
    # Format output based on input types
    if single1 and single2:
        return similarities[0, 0]
    elif single1:
        return similarities[0, :]
    elif single2:
        return similarities[:, 0]
    else:
        return similarities

def hedjazi_similarity(interval1, interval2):
    """
    Calculate the marginal similarity (generalized Jaccard) between two intervals (1D or multi-D).

    Parameters:
        interval1: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples1, n_dims, 2)
        interval2: Single interval [min, max], multi-dimensional array of shape (n_dims, 2),
                  or batch of intervals with shape (n_samples2, n_dims, 2)

    Returns:
        float or np.ndarray: Marginal similarity/similarities
                           - scalar if both inputs are single intervals
                           - vector if one input is batch
                           - matrix if both inputs are batches
    """
    # Convert inputs to numpy arrays
    a = np.asarray(interval1, dtype=np.float64)
    b = np.asarray(interval2, dtype=np.float64)
    
    # Handle different input dimensions
    single1 = a.ndim <= 2
    single2 = b.ndim <= 2
    
    # Legacy compatibility for simple cases
    if a.ndim <= 2 and b.ndim <= 2:
        # Promote 1D intervals to shape (n_dims, 2)
        if a.ndim == 1:
            a = a[np.newaxis, :]
            b = b[np.newaxis, :]
        
        # Split into lower and upper bounds
        lower_a, upper_a = a[:, 0], a[:, 1]
        lower_b, upper_b = b[:, 0], b[:, 1]
        
        # Compute intersection length for each dimension
        intersection = np.maximum(0, np.minimum(upper_a, upper_b) - np.maximum(lower_a, lower_b))
        
        # Compute union length for each dimension
        length_a = upper_a - lower_a
        length_b = upper_b - lower_b
        union = length_a + length_b - intersection
        
        # Compute gap distance (non-overlap) for each dimension
        distance = np.maximum(0, np.maximum(lower_a, lower_b) - np.minimum(upper_a, upper_b))
        
        # Compute overall domain span for each dimension
        domain = np.maximum(upper_a, upper_b) - np.minimum(lower_a, lower_b)
        
        # Calculate marginal similarity per dimension; guard against zero division
        sim = np.where(
            (union > 0) & (domain > 0),
            0.5 * (intersection / union + 1 - distance / domain),
            0.0
        )
        
        # Return the average similarity across all dimensions
        return sim.mean()
    
    # Standardize to 3D format: (n_samples, n_dims, 2)
    if a.ndim == 2:
        a = a[None, :, :]
    if b.ndim == 2:
        b = b[None, :, :]
    
    # Vectorized computation using broadcasting
    # Extract bounds: (n_samples1, 1, n_dims) and (1, n_samples2, n_dims)
    lower_a, upper_a = a[:, None, :, 0], a[:, None, :, 1]
    lower_b, upper_b = b[None, :, :, 0], b[None, :, :, 1]
    
    # Compute intersection bounds: (n_samples1, n_samples2, n_dims)
    intersection_lower = np.maximum(lower_a, lower_b)
    intersection_upper = np.minimum(upper_a, upper_b)
    intersection = np.maximum(0, intersection_upper - intersection_lower)
    
    # Compute union length for each dimension: (n_samples1, n_samples2, n_dims)
    length_a = upper_a - lower_a
    length_b = upper_b - lower_b
    union = length_a + length_b - intersection
    
    # Compute gap distance (non-overlap) for each dimension
    distance = np.maximum(0, np.maximum(lower_a, lower_b) - np.minimum(upper_a, upper_b))
    
    # Compute overall domain span for each dimension
    domain = np.maximum(upper_a, upper_b) - np.minimum(lower_a, lower_b)
    
    # Calculate marginal similarity per dimension; guard against zero division
    sim_per_dim = np.where(
        (union > 0) & (domain > 0),
        0.5 * (intersection / union + 1 - distance / domain),
        0.0
    )
    
    # Average across dimensions: (n_samples1, n_samples2)
    similarities = np.mean(sim_per_dim, axis=2)
    
    # Format output based on input types
    if single1 and single2:
        return similarities[0, 0]
    elif single1:
        return similarities[0, :]
    elif single2:
        return similarities[:, 0]
    else:
        return similarities

# Dictionary mapping similarity type to similarity function
SIMILARITY_FUNCTIONS = {
    "jaccard": jaccard_similarity,
    "dice": dice_similarity,
    "bidirectional_min": bidirectional_similarity_min,
    "bidirectional_prod": bidirectional_similarity_prod,
    "hedjazi": hedjazi_similarity,
}

# Convenience function to convert similarities to distances
def similarity_to_distance(similarity_func):
    """
    Convert a similarity function to a distance function.
    Distance = 1 - Similarity
    
    Parameters:
        similarity_func: A similarity function from SIMILARITY_FUNCTIONS
        
    Returns:
        function: A distance function with the same signature
    """
    def distance_wrapper(interval1, interval2):
        similarity = similarity_func(interval1, interval2)
        return 1.0 - similarity
    
    return distance_wrapper

# Distance versions of similarity functions
SIMILARITY_DISTANCE_FUNCTIONS = {
    "jaccard_distance": similarity_to_distance(jaccard_similarity),
    "dice_distance": similarity_to_distance(dice_similarity),
    "bidirectional_min_distance": similarity_to_distance(bidirectional_similarity_min),
    "bidirectional_prod_distance": similarity_to_distance(bidirectional_similarity_prod),
    "hedjazi_distance": similarity_to_distance(hedjazi_similarity),
}


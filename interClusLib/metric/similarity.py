import numpy as np

def jaccard_similarity(interval1, interval2):
    """
    Calculate the Jaccard similarity between two intervals.
    
    Parameters:
        interval1: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        interval2: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        
    Returns:
        float: Jaccard similarity
    """
    # Check if inputs are multi-dimensional
    if len(interval1.shape) > 1:
        n_dims = interval1.shape[0]
        similarities = []
        for d in range(n_dims):
            # Calculate 1D Jaccard for each dimension
            intersection = max(0, min(interval1[d, 1], interval2[d, 1]) - max(interval1[d, 0], interval2[d, 0]))
            if intersection > 0:
                union = (interval1[d, 1] - interval1[d, 0]) + (interval2[d, 1] - interval2[d, 0]) - intersection
            else:
                union = (interval1[d, 1] - interval1[d, 0]) + (interval2[d, 1] - interval2[d, 0])
            sim = intersection / union if union > 0 else 0
            similarities.append(sim)
        return np.mean(similarities)
    else:
        # Single-dimensional case
        intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
        if intersection > 0:
            union = (interval1[1] - interval1[0]) + (interval2[1] - interval2[0]) - intersection
        else:
            union = (interval1[1] - interval1[0]) + (interval2[1] - interval2[0])
        return intersection / union if union > 0 else 0

def dice_similarity(interval1, interval2):
    """
    Calculate the Dice similarity between two intervals.
    
    Parameters:
        interval1: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        interval2: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        
    Returns:
        float: Dice similarity
    """
    # Check if inputs are multi-dimensional
    if len(interval1.shape) > 1:
        n_dims = interval1.shape[0]
        similarities = []
        for d in range(n_dims):
            # Calculate 1D Dice for each dimension
            intersection = max(0, min(interval1[d, 1], interval2[d, 1]) - max(interval1[d, 0], interval2[d, 0]))
            sum_intervals = (interval1[d, 1] - interval1[d, 0]) + (interval2[d, 1] - interval2[d, 0])
            sim = 2 * intersection / sum_intervals if sum_intervals > 0 else 0
            similarities.append(sim)
        return np.mean(similarities)
    else:
        # Single-dimensional case
        intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
        sum_intervals = (interval1[1] - interval1[0]) + (interval2[1] - interval2[0])
        return 2 * intersection / sum_intervals if sum_intervals > 0 else 0

def bidirectional_similarity_min(interval1, interval2):
    """
    Calculate the minimum bidirectional subset similarity between two intervals.
    
    Parameters:
        interval1: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        interval2: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        
    Returns:
        float: Minimum bidirectional subset similarity
    """
    # Check if inputs are multi-dimensional
    if len(interval1.shape) > 1:
        n_dims = interval1.shape[0]
        similarities = []
        for d in range(n_dims):
            # Calculate 1D bidirectional min for each dimension
            intersection = max(0, min(interval1[d, 1], interval2[d, 1]) - max(interval1[d, 0], interval2[d, 0]))
            non_overlap_a_b = max(0, (interval1[d, 1] - interval1[d, 0]) - intersection)
            non_overlap_b_a = max(0, (interval2[d, 1] - interval2[d, 0]) - intersection)
            
            if intersection + non_overlap_a_b > 0 and intersection + non_overlap_b_a > 0:
                reciprocal_a_b = intersection / (intersection + non_overlap_a_b)
                reciprocal_b_a = intersection / (intersection + non_overlap_b_a)
                sim = min(reciprocal_a_b, reciprocal_b_a)
            else:
                sim = 0
            
            similarities.append(sim)
        return np.mean(similarities)
    else:
        # Single-dimensional case
        intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
        non_overlap_a_b = max(0, (interval1[1] - interval1[0]) - intersection)
        non_overlap_b_a = max(0, (interval2[1] - interval2[0]) - intersection)
        
        if intersection + non_overlap_a_b > 0 and intersection + non_overlap_b_a > 0:
            reciprocal_a_b = intersection / (intersection + non_overlap_a_b)
            reciprocal_b_a = intersection / (intersection + non_overlap_b_a)
            return min(reciprocal_a_b, reciprocal_b_a)
        else:
            return 0

def bidirectional_similarity_prod(interval1, interval2):
    """
    Calculate the product bidirectional subset similarity between two intervals.
    
    Parameters:
        interval1: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        interval2: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        
    Returns:
        float: Product bidirectional subset similarity
    """
    # Check if inputs are multi-dimensional
    if len(interval1.shape) > 1:
        n_dims = interval1.shape[0]
        similarities = []
        for d in range(n_dims):
            # Calculate 1D bidirectional prod for each dimension
            intersection = max(0, min(interval1[d, 1], interval2[d, 1]) - max(interval1[d, 0], interval2[d, 0]))
            non_overlap_a_b = max(0, (interval1[d, 1] - interval1[d, 0]) - intersection)
            non_overlap_b_a = max(0, (interval2[d, 1] - interval2[d, 0]) - intersection)
            
            if intersection + non_overlap_a_b > 0 and intersection + non_overlap_b_a > 0:
                reciprocal_a_b = intersection / (intersection + non_overlap_a_b)
                reciprocal_b_a = intersection / (intersection + non_overlap_b_a)
                sim = reciprocal_a_b * reciprocal_b_a
            else:
                sim = 0
            
            similarities.append(sim)
        return np.mean(similarities)
    else:
        # Single-dimensional case
        intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
        non_overlap_a_b = max(0, (interval1[1] - interval1[0]) - intersection)
        non_overlap_b_a = max(0, (interval2[1] - interval2[0]) - intersection)
        
        if intersection + non_overlap_a_b > 0 and intersection + non_overlap_b_a > 0:
            reciprocal_a_b = intersection / (intersection + non_overlap_a_b)
            reciprocal_b_a = intersection / (intersection + non_overlap_b_a)
            return reciprocal_a_b * reciprocal_b_a
        else:
            return 0

def marginal_similarity(interval1, interval2):
    """
    Calculate the marginal similarity (generalized Jaccard) between two intervals.
    
    Parameters:
        interval1: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        interval2: Single interval [min, max] or multi-dimensional array of shape (n_dims, 2)
        
    Returns:
        float: Marginal similarity
    """
    # Check if inputs are multi-dimensional
    if len(interval1.shape) > 1:
        n_dims = interval1.shape[0]
        similarities = []
        for d in range(n_dims):
            # Calculate 1D marginal similarity for each dimension
            intersection = max(0, min(interval1[d, 1], interval2[d, 1]) - max(interval1[d, 0], interval2[d, 0]))
            union = (interval1[d, 1] - interval1[d, 0]) + (interval2[d, 1] - interval2[d, 0]) - intersection
            distance = max(0, max(interval1[d, 0], interval2[d, 0]) - min(interval1[d, 1], interval2[d, 1]))
            domain = max(interval1[d, 1], interval2[d, 1]) - min(interval1[d, 0], interval2[d, 0])
            
            if union > 0 and domain > 0:
                sim = 0.5 * (intersection / union + 1 - distance / domain)
            else:
                sim = 0
            
            similarities.append(sim)
        return np.mean(similarities)
    else:
        # Single-dimensional case
        intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
        union = (interval1[1] - interval1[0]) + (interval2[1] - interval2[0]) - intersection
        distance = max(0, max(interval1[0], interval2[0]) - min(interval1[1], interval2[1]))
        domain = max(interval1[1], interval2[1]) - min(interval1[0], interval2[0])
        
        if union > 0 and domain > 0:
            return 0.5 * (intersection / union + 1 - distance / domain)
        else:
            return 0

# Dictionary mapping similarity type to similarity function
SIMILARITY_FUNCTIONS = {
    "jaccard": jaccard_similarity,
    "dice": dice_similarity,
    "bidirectional_min": bidirectional_similarity_min,
    "bidirectional_prod": bidirectional_similarity_prod,
    "marginal": marginal_similarity,
}


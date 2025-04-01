import numpy as np

def fix_inverted_intervals(interval_data, method='swap', inverted_mask=None):
    """Fix intervals where lower bound is greater than upper bound.
    
    Parameters:
    -----------
    interval_data: numpy.ndarray
        Interval data with shape (n_samples, n_dim, 2) where [:,:,0] is lower bound
        and [:,:,1] is upper bound
    method: str, default='swap'
        Method to use for fixing inverted intervals:
        - 'swap': Swap lower and upper bounds
        - 'remove': Remove samples containing inverted intervals
    inverted_mask: numpy.ndarray, optional
        Boolean mask indicating which intervals are inverted (n_samples, n_dim)
        If None, automatically generated based on lower > upper condition
        
    Returns:
    --------
    numpy.ndarray
        Fixed interval data with shape (n_samples, n_dim, 2) if method is not 'remove'
        or (n_valid_samples, n_dim, 2) if method is 'remove'
    numpy.ndarray
        Boolean mask indicating which intervals were inverted (n_samples, n_dim)
    """
    n_samples, n_dim, _ = interval_data.shape
    fixed_data = interval_data.copy()
    
    # Create a mask for intervals where either bound is None
    lower_is_none = np.array([[x is None for x in row[:,0]] for row in interval_data])
    upper_is_none = np.array([[x is None for x in row[:,1]] for row in interval_data])
    has_none = np.logical_or(lower_is_none, upper_is_none)
    
    # If no inverted_mask provided, identify inverted intervals (lower > upper)
    if inverted_mask is None:
        # Create a mask for inverted intervals, but set False where any bound is None
        inverted_mask = np.zeros((n_samples, n_dim), dtype=bool)
        for i in range(n_samples):
            for j in range(n_dim):
                if not has_none[i, j]:  # Only compare non-None values
                    inverted_mask[i, j] = interval_data[i, j, 0] > interval_data[i, j, 1]
    
    # If no inverted intervals, return original data
    if not np.any(inverted_mask):
        return fixed_data, inverted_mask
    
    # If method is 'remove', remove samples with inverted intervals
    if method == 'remove':
        # Find samples that have at least one inverted interval
        samples_to_remove = np.any(inverted_mask, axis=1)
        # Keep only samples without inverted intervals
        return interval_data[~samples_to_remove], inverted_mask
    
    # Fix inverted intervals based on chosen method
    for d in range(n_dim):
        for i in range(n_samples):
            # Only swap if the interval is inverted AND neither bound is None
            if inverted_mask[i, d] and not has_none[i, d]:
                if method == 'swap':
                    # Swap lower and upper bounds
                    temp = fixed_data[i, d, 0]
                    fixed_data[i, d, 0] = fixed_data[i, d, 1]
                    fixed_data[i, d, 1] = temp
    
    return fixed_data, inverted_mask
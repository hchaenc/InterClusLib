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
        - 'mean': Replace with the mean of the two bounds
        - 'min_max': Use smaller value as lower bound and larger as upper bound
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
    
    # If no inverted_mask provided, identify inverted intervals (lower > upper)
    if inverted_mask is None:
        inverted_mask = interval_data[:,:,0] > interval_data[:,:,1]
    
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
        inverted_idx = inverted_mask[:, d]
        if not np.any(inverted_idx):
            continue
            
        if method == 'swap':
            # Swap lower and upper bounds
            temp = fixed_data[inverted_idx, d, 0].copy()
            fixed_data[inverted_idx, d, 0] = fixed_data[inverted_idx, d, 1]
            fixed_data[inverted_idx, d, 1] = temp
    
    return fixed_data, inverted_mask
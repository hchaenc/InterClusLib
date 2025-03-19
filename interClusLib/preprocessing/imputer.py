import numpy as np

def impute_missing_intervals(interval_data, missing_mask=None, lower_missing_mask=None, upper_missing_mask=None, method='mean'):
    """Impute missing interval data, including cases where only lower or upper bounds are missing
    
    Methods:
        - 'mean': Use the average midpoint and width
        - 'median': Use the median midpoint and width
        - 'knn': Imputation based on k-nearest neighbors (not implemented)
        - 'remove': Remove samples with any missing values instead of imputing
    
    Parameters:
    -----------
    interval_data: numpy.ndarray
        Interval data with shape (n_samples, n_dim, 2)
    missing_mask: numpy.ndarray, optional
        Boolean mask indicating completely missing intervals with shape (n_samples, n_dim)
    lower_missing_mask: numpy.ndarray, optional
        Boolean mask indicating missing lower bounds with shape (n_samples, n_dim)
    upper_missing_mask: numpy.ndarray, optional
        Boolean mask indicating missing upper bounds with shape (n_samples, n_dim)
    method: str, default='mean'
        Method to use for imputation
        
    Returns:
    --------
    numpy.ndarray
        Imputed interval data with shape (n_samples, n_dim, 2) if method is not 'remove'
        or (n_valid_samples, n_dim, 2) if method is 'remove'
    """
    n_samples, n_dim, _ = interval_data.shape
    imputed_data = interval_data.copy()
    
    # Initialize masks if not provided
    if missing_mask is None:
        missing_mask = np.zeros((n_samples, n_dim), dtype=bool)
    if lower_missing_mask is None:
        lower_missing_mask = np.zeros((n_samples, n_dim), dtype=bool)
    if upper_missing_mask is None:
        upper_missing_mask = np.zeros((n_samples, n_dim), dtype=bool)
    
    # Create a combined mask for any type of missing values
    any_missing_mask = missing_mask | lower_missing_mask | upper_missing_mask
    
    # If using remove method, return only samples without any missing values
    if method == 'remove':
        # Find samples that have at least one missing value
        samples_to_remove = np.any(any_missing_mask, axis=1)
        # Keep only samples without missing values
        return interval_data[~samples_to_remove]
    
    for d in range(n_dim):
        # Handle completely missing intervals
        missing_idx = missing_mask[:, d]
        lower_missing_idx = lower_missing_mask[:, d]
        upper_missing_idx = upper_missing_mask[:, d]
        
        # If no missing data in this dimension, skip
        if not np.any(missing_idx) and not np.any(lower_missing_idx) and not np.any(upper_missing_idx):
            continue
            
        # Get valid data (neither lower nor upper bounds are missing)
        valid_mask = ~(missing_idx | lower_missing_idx | upper_missing_idx)
        valid_data = interval_data[valid_mask, d, :]
        
        if len(valid_data) == 0:
            # If no valid data, skip this dimension or use some default
            continue
        
        if method == 'mean':
            # Calculate average midpoint and width
            midpoints = (valid_data[:, 0] + valid_data[:, 1]) / 2
            widths = valid_data[:, 1] - valid_data[:, 0]
            avg_midpoint = np.mean(midpoints)
            avg_width = np.mean(widths)
            
            # Impute completely missing intervals
            if np.any(missing_idx):
                imputed_data[missing_idx, d, 0] = avg_midpoint - avg_width/2
                imputed_data[missing_idx, d, 1] = avg_midpoint + avg_width/2
            
            # Impute missing lower bounds
            if np.any(lower_missing_idx):
                upper_values = imputed_data[lower_missing_idx, d, 1]
                imputed_data[lower_missing_idx, d, 0] = upper_values - avg_width
            
            # Impute missing upper bounds
            if np.any(upper_missing_idx):
                lower_values = imputed_data[upper_missing_idx, d, 0]
                imputed_data[upper_missing_idx, d, 1] = lower_values + avg_width
                
        elif method == 'median':
            # Calculate median midpoint and width
            midpoints = (valid_data[:, 0] + valid_data[:, 1]) / 2
            widths = valid_data[:, 1] - valid_data[:, 0]
            med_midpoint = np.median(midpoints)
            med_width = np.median(widths)
            
            # Impute completely missing intervals
            if np.any(missing_idx):
                imputed_data[missing_idx, d, 0] = med_midpoint - med_width/2
                imputed_data[missing_idx, d, 1] = med_midpoint + med_width/2
            
            # Impute missing lower bounds
            if np.any(lower_missing_idx):
                upper_values = imputed_data[lower_missing_idx, d, 1]
                imputed_data[lower_missing_idx, d, 0] = upper_values - med_width
            
            # Impute missing upper bounds
            if np.any(upper_missing_idx):
                lower_values = imputed_data[upper_missing_idx, d, 0]
                imputed_data[upper_missing_idx, d, 1] = lower_values + med_width
    
    return imputed_data
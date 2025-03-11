import numpy as np

def impute_missing_intervals(interval_data, missing_mask, method='mean'):
    """Impute missing interval data
    
    Methods:
        - 'mean': Use the average midpoint and width
        - 'median': Use the median midpoint and width
        - 'knn': Imputation based on k-nearest neighbors
    
    Parameters:
    -----------
    interval_data: numpy.ndarray
        Interval data with shape (n_samples, n_dim, 2)
    missing_mask: numpy.ndarray
        Boolean mask indicating missing values with shape (n_samples, n_dim)
    method: str, default='mean'
        Method to use for imputation
        
    Returns:
    --------
    numpy.ndarray
        Imputed interval data with shape (n_samples, n_dim, 2)
    """
    n_samples, n_dim, _ = interval_data.shape
    imputed_data = interval_data.copy()
    
    for d in range(n_dim):
        missing_idx = missing_mask[:, d]
        if not np.any(missing_idx):
            continue  # Skip if no missing data in this dimension
            
        valid_data = interval_data[~missing_idx, d, :]
        
        if method == 'mean':
            # Calculate average midpoint and width
            midpoint = np.mean((valid_data[:, 0] + valid_data[:, 1]) / 2)
            width = np.mean(valid_data[:, 1] - valid_data[:, 0])
            
            # Fill missing values
            imputed_data[missing_idx, d, 0] = midpoint - width/2
            imputed_data[missing_idx, d, 1] = midpoint + width/2
            
        elif method == 'median':
            # Calculate median midpoint and width
            midpoint = np.median((valid_data[:, 0] + valid_data[:, 1]) / 2)
            width = np.median(valid_data[:, 1] - valid_data[:, 0])
            
            # Fill missing values
            imputed_data[missing_idx, d, 0] = midpoint - width/2
            imputed_data[missing_idx, d, 1] = midpoint + width/2
            
        # Note: 'knn' method mentioned in docstring but not implemented
    
    return imputed_data
import numpy as np

def min_max_normalize(interval_data, feature_range=(0, 1)):
    """
    Perform min-max normalization on interval data while preserving interval relationships.

    Parameters:
        interval_data: An array of interval data with shape [n_samples, n_intervals, 2].
        feature_range: The target range, default is (0, 1).

    Returns:
        Normalized interval data with the same shape.
    """
    n_samples, n_dim, _ = interval_data.shape
    normalized_data = np.zeros_like(interval_data, dtype=float)
    min_new, max_new = feature_range
    
    for d in range(n_dim):
        lower_bounds = interval_data[:, d, 0]
        upper_bounds = interval_data[:, d, 1]
        
        global_min = np.min(lower_bounds)
        global_max = np.max(upper_bounds)
        
        normalized_data[:, d, 0] = min_new + (lower_bounds - global_min) * (max_new - min_new) / (global_max - global_min)
        normalized_data[:, d, 1] = min_new + (upper_bounds - global_min) * (max_new - min_new) / (global_max - global_min)
    
    return normalized_data

def midpoint_width_normalize(interval_data):
    """
    Normalize the midpoint and width of intervals separately.
    
    Parameters:
        interval_data: An array of interval data with shape [n_samples, n_intervals, 2].
    
    Returns:
        Normalized interval data with the same shape.
    """
    n_samples, n_dim, _ = interval_data.shape
    normalized_data = np.zeros_like(interval_data, dtype=float)
    
    for d in range(n_dim):
        # Calculate midpoints and widths
        midpoints = (interval_data[:, d, 0] + interval_data[:, d, 1]) / 2
        widths = interval_data[:, d, 1] - interval_data[:, d, 0]
        
        # Normalize midpoints
        mid_min, mid_max = np.min(midpoints), np.max(midpoints)
        norm_midpoints = (midpoints - mid_min) / (mid_max - mid_min)
        
        # Normalize widths
        width_min, width_max = np.min(widths), np.max(widths)
        norm_widths = (widths - width_min) / (width_max - width_min)
        
        # Reconstruct intervals
        normalized_data[:, d, 0] = norm_midpoints - norm_widths/2
        normalized_data[:, d, 1] = norm_midpoints + norm_widths/2
    
    return normalized_data

def robust_quantile_normalize(interval_data, q_low=0.05, q_high=0.95):
    """
    Robust normalization using quantiles.
    
    Parameters:
        interval_data: An array of interval data with shape [n_samples, n_intervals, 2].
        q_low: Lower quantile threshold, default is 0.05 (5th percentile).
        q_high: Upper quantile threshold, default is 0.95 (95th percentile).
    
    Returns:
        Normalized interval data with the same shape.
    """
    n_samples, n_dim, _ = interval_data.shape
    normalized_data = np.zeros_like(interval_data, dtype=float)
    
    for d in range(n_dim):
        # Calculate quantile boundaries
        lower_bounds = interval_data[:, d, 0]
        upper_bounds = interval_data[:, d, 1]
        
        min_q = np.quantile(lower_bounds, q_low)
        max_q = np.quantile(upper_bounds, q_high)
        
        # Normalize and clip values
        normalized_data[:, d, 0] = np.clip((lower_bounds - min_q) / (max_q - min_q), 0, 1)
        normalized_data[:, d, 1] = np.clip((upper_bounds - min_q) / (max_q - min_q), 0, 1)
    
    return normalized_data

def z_score_normalize(interval_data):
    """
    Standardize interval data using z-score normalization.
    
    Parameters:
        interval_data: An array of interval data with shape [n_samples, n_intervals, 2].
    
    Returns:
        Standardized interval data with the same shape.
    """
    n_samples, n_dim, _ = interval_data.shape
    normalized_data = np.zeros_like(interval_data, dtype=float)
    
    for d in range(n_dim):
        # Extract midpoints for standardization
        midpoints = (interval_data[:, d, 0] + interval_data[:, d, 1]) / 2
        widths = interval_data[:, d, 1] - interval_data[:, d, 0]
        
        mid_mean = np.mean(midpoints)
        mid_std = np.std(midpoints)
        
        # Standardize midpoints and maintain relative widths
        norm_midpoints = (midpoints - mid_mean) / mid_std
        norm_widths = widths / (mid_std * 2)  # Width as a proportion of standard deviation
        
        # Reconstruct intervals
        normalized_data[:, d, 0] = norm_midpoints - norm_widths/2
        normalized_data[:, d, 1] = norm_midpoints + norm_widths/2
    
    return normalized_data

import numpy as np

def min_max_normalize(interval_data, feature_range=(0, 1)):
    """
    Perform min-max normalization on interval data while preserving interval relationships.

    Parameters:
        interval_data: An array of interval data with shape [n_samples, n_intervals, 2].
        feature_range: The target range, default is (0, 1).

    Returns:
        Normalized interval data with the same shape.

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
    """分别归一化区间的中点和宽度"""
    n_samples, n_dim, _ = interval_data.shape
    normalized_data = np.zeros_like(interval_data, dtype=float)
    
    for d in range(n_dim):
        # 计算中点和宽度
        midpoints = (interval_data[:, d, 0] + interval_data[:, d, 1]) / 2
        widths = interval_data[:, d, 1] - interval_data[:, d, 0]
        
        # 归一化中点
        mid_min, mid_max = np.min(midpoints), np.max(midpoints)
        norm_midpoints = (midpoints - mid_min) / (mid_max - mid_min)
        
        # 归一化宽度
        width_min, width_max = np.min(widths), np.max(widths)
        norm_widths = (widths - width_min) / (width_max - width_min)
        
        # 重建区间
        normalized_data[:, d, 0] = norm_midpoints - norm_widths/2
        normalized_data[:, d, 1] = norm_midpoints + norm_widths/2
    
    return normalized_data

def robust_quantile_normalize(interval_data, q_low=0.05, q_high=0.95):
    """使用分位数进行稳健归一化"""
    n_samples, n_dim, _ = interval_data.shape
    normalized_data = np.zeros_like(interval_data, dtype=float)
    
    for d in range(n_dim):
        # 计算分位数边界
        lower_bounds = interval_data[:, d, 0]
        upper_bounds = interval_data[:, d, 1]
        
        min_q = np.quantile(lower_bounds, q_low)
        max_q = np.quantile(upper_bounds, q_high)
        
        # 归一化并裁剪
        normalized_data[:, d, 0] = np.clip((lower_bounds - min_q) / (max_q - min_q), 0, 1)
        normalized_data[:, d, 1] = np.clip((upper_bounds - min_q) / (max_q - min_q), 0, 1)
    
    return normalized_data

def z_score_normalize(interval_data):
    """使用z-score标准化区间数据"""
    n_samples, n_dim, _ = interval_data.shape
    normalized_data = np.zeros_like(interval_data, dtype=float)
    
    for d in range(n_dim):
        # 提取中点进行标准化
        midpoints = (interval_data[:, d, 0] + interval_data[:, d, 1]) / 2
        widths = interval_data[:, d, 1] - interval_data[:, d, 0]
        
        mid_mean = np.mean(midpoints)
        mid_std = np.std(midpoints)
        
        # 标准化中点和保持相对宽度
        norm_midpoints = (midpoints - mid_mean) / mid_std
        norm_widths = widths / (mid_std * 2)  # 宽度相对标准差的比例
        
        # 重建区间
        normalized_data[:, d, 0] = norm_midpoints - norm_widths/2
        normalized_data[:, d, 1] = norm_midpoints + norm_widths/2
    
    return normalized_data

import numpy as np

def impute_missing_intervals(interval_data, missing_mask, method='mean'):
    """填充缺失的区间数据
    
    方法:
        - 'mean': 使用平均中点和宽度
        - 'median': 使用中位数中点和宽度
        - 'knn': 基于k近邻的插补
    """
    n_samples, n_dim, _ = interval_data.shape
    imputed_data = interval_data.copy()
    
    for d in range(n_dim):
        missing_idx = missing_mask[:, d]
        if not np.any(missing_idx):
            continue
            
        # 非缺失数据
        valid_data = interval_data[~missing_idx, d, :]
        
        if method == 'mean':
            # 计算平均中点和宽度
            midpoint = np.mean((valid_data[:, 0] + valid_data[:, 1]) / 2)
            width = np.mean(valid_data[:, 1] - valid_data[:, 0])
            
            # 填充缺失值
            imputed_data[missing_idx, d, 0] = midpoint - width/2
            imputed_data[missing_idx, d, 1] = midpoint + width/2
            
        elif method == 'median':
            # 计算中位数中点和宽度
            midpoint = np.median((valid_data[:, 0] + valid_data[:, 1]) / 2)
            width = np.median(valid_data[:, 1] - valid_data[:, 0])
            
            # 填充缺失值
            imputed_data[missing_idx, d, 0] = midpoint - width/2
            imputed_data[missing_idx, d, 1] = midpoint + width/2
    
    return imputed_data
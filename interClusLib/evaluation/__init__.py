from .CalinskiHarabaszIndex import calinski_harabasz_index
from .DavisBouldinIndex import davies_bouldin_index
from .DistortionScore import distortion_score
from .DunnIndex import dunn_index
from .SilhouetteScore import silhouette_score

# 评估指标映射字典
EVALUATION = {
    # 聚类评估指标
    'distortion': distortion_score,         # 失真度/惯性
    'silhouette': silhouette_score,         # 轮廓系数
    'calinski_harabasz': calinski_harabasz_index,  # Calinski-Harabasz指数
    'davies_bouldin': davies_bouldin_index, # Davies-Bouldin指数
    'dunn': dunn_index,                     # Dunn指数
    
    # 常用别名
    'inertia': distortion_score,      # 惯性(别名)
    'ch_index': calinski_harabasz_index,   # CH指数(别名) 
    'db_index': davies_bouldin_index,      # DB指数(别名)
}

# 指标优化方向字典 (True表示值越大越好，False表示值越小越好)
EVALUATION_MAXIMIZE = {
    'distortion': False,          # 越小越好
    'silhouette': True,           # 越大越好
    'calinski_harabasz': True,    # 越大越好
    'davies_bouldin': False,      # 越小越好
    'dunn': True,                 # 越大越好
    'inertia': False,             # 越小越好
    'ch_index': True,             # 越大越好
    'db_index': False,            # 越小越好
}

# 指标描述字典
EVALUATION_DESCRIPTIONS = {
    'distortion': 'Distortion Score (lower is better)',
    'silhouette': 'Silhouette Coefficient (higher is better)',
    'calinski_harabasz': 'Calinski-Harabasz Index (higher is better)',
    'davies_bouldin': 'Davies-Bouldin Index (lower is better)',
    'dunn': 'Dunn Index (higher is better)',
    'inertia': 'Inertia (lower is better)',
    'ch_index': 'Calinski-Harabasz Index (higher is better)',
    'db_index': 'Davies-Bouldin Index (lower is better)'
}

__all__ = ['calinski_harabasz_index', 'davies_bouldin_index', 'distortion_score', 'dunn_index', 'silhouette_score']
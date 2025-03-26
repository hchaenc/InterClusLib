"""
Base module for cluster number evaluation methods.
"""
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class ClusterEvaluationMethod(ABC):
    """Abstract base class for cluster number evaluation methods."""
    
    def __init__(self, min_clusters=2, max_clusters=20, **kwargs):
        """
        Initialize the evaluation method.
        
        Parameters:
        min_clusters: int, minimum number of clusters to consider
        max_clusters: int, maximum number of clusters to consider
        **kwargs: additional parameters
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.eval_results = None
        self.optimal_k = None
        self.kwargs = kwargs
    
    def _validate_and_format_data(self, eval_data):
        """
        Validate and format evaluation data.
        
        Parameters:
        eval_data: dict or array, containing cluster counts and corresponding evaluation metrics
        
        Returns:
        numpy.ndarray: formatted evaluation data with shape (n, 2)
        """
        # Convert input data format
        if isinstance(eval_data, dict):
            # 确保键被转换为整数
            data = np.array([(int(k), float(v)) for k, v in sorted(eval_data.items(), key=lambda x: int(x[0]))])
        else:
            # 确保数据的第一列（聚类数）是整数类型
            data = np.array(eval_data, dtype=float)
            
        # Ensure data is sorted by cluster number
        data = data[data[:, 0].argsort()]
        
        # Apply cluster count limits
        mask = (data[:, 0] >= self.min_clusters) & (data[:, 0] <= self.max_clusters)
        data = data[mask]
        
        if len(data) < 3:
            import warnings
            warnings.warn("Warning: Insufficient data points. The evaluation method may not produce reliable results.")
            
        return data
    
    @abstractmethod
    def evaluate(self, eval_data):
        """
        Evaluate the optimal number of clusters.
        
        Parameters:
        eval_data: evaluation data
        
        Returns:
        int: optimal number of clusters
        """
        pass
    
    def plot(self, title=None, show_optimal=True, figsize=(10, 6), 
             xlabel='Number of Clusters', ylabel='Evaluation Metric'):
        """
        Plot the evaluation graph and the determined optimal number of clusters.
        
        Parameters:
        title: str, chart title
        show_optimal: bool, whether to display the optimal number of clusters
        figsize: tuple, figure size
        xlabel: str, x-axis label
        ylabel: str, y-axis label
        
        Returns:
        plt: matplotlib figure object
        """
        if self.eval_results is None:
            raise ValueError("Must run evaluate method first to determine the optimal number of clusters")
        
        plt.figure(figsize=figsize)
        
        x = self.eval_results[:, 0]
        y = self.eval_results[:, 1]
        
        plt.plot(x, y, 'bo-', linewidth=2, markersize=8)
        plt.grid(True, alpha=0.3)
        
        if show_optimal and self.optimal_k is not None:
            plt.axvline(x=self.optimal_k, color='red', linestyle='--', 
                      label=f'Optimal k={self.optimal_k}')
            plt.legend()
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title or f'{self.__class__.__name__} - Cluster Evaluation')
        
        return plt
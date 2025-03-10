import numpy as np
from .base_evaluator import ClusterEvaluationMethod
from interClusLib.evaluation import EVALUATION_MAXIMIZE

class MaxMinClusterEvaluator(ClusterEvaluationMethod):
    """
    Cluster evaluator that determines the optimal number of clusters
    by finding either the maximum or minimum value of the evaluation metric.
    
    This is useful for metrics where either the maximum value (e.g., silhouette score)
    or the minimum value (e.g., inertia, Davies-Bouldin index) indicates the optimal
    clustering.
    """
    
    def __init__(self, min_clusters=2, max_clusters=20, find_max=True, metric=None, **kwargs):
        """
        Initialize the Max/Min cluster evaluator.
        
        Parameters:
        min_clusters: int, minimum number of clusters to consider
        max_clusters: int, maximum number of clusters to consider
        find_max: bool, if True, looks for maximum value as optimal;
                  if False, looks for minimum value
        metric: str, name of the evaluation metric (e.g., 'silhouette', 'inertia')
               If provided, this will override the find_max parameter based on the 
               optimization direction of the metric defined in EVALUATION_MAXIMIZE
        **kwargs: additional parameters passed to the parent class
        """
        super().__init__(min_clusters=min_clusters, max_clusters=max_clusters, **kwargs)
        
        # If metric is provided, use its optimization direction from the dictionary
        if metric is not None and metric in EVALUATION_MAXIMIZE:
            self.find_max = EVALUATION_MAXIMIZE[metric]
            self.metric_name = metric
        else:
            self.find_max = find_max
            self.metric_name = None
    
    def evaluate(self, eval_data):
        """
        Evaluate the optimal number of clusters based on finding the maximum or
        minimum value in the evaluation data.
        
        Parameters:
        eval_data: dict or array, containing cluster counts and corresponding evaluation metrics
        
        Returns:
        int: optimal number of clusters
        """
        # Validate and format input data
        self.eval_results = self._validate_and_format_data(eval_data)
        
        if len(self.eval_results) == 0:
            raise ValueError("No valid data points within the specified cluster range")
        
        # Find the index of the max or min value
        if self.find_max:
            optimal_idx = np.argmax(self.eval_results[:, 1])
        else:
            optimal_idx = np.argmin(self.eval_results[:, 1])
        
        # Get the corresponding number of clusters
        self.optimal_k = int(self.eval_results[optimal_idx, 0])
        
        return self.optimal_k
    
    def plot(self, title=None, show_optimal=True, figsize=(10, 6),
             xlabel='Number of Clusters', ylabel='Evaluation Metric'):
        """
        Plot the evaluation results with the optimal cluster number highlighted.
        
        Parameters:
        title: str, chart title (default is generated based on class name and optimization type)
        show_optimal: bool, whether to display the optimal number of clusters
        figsize: tuple, figure size
        xlabel: str, x-axis label
        ylabel: str, y-axis label
        
        Returns:
        plt: matplotlib figure object
        """
        # Generate appropriate title if not provided
        if title is None:
            optimization_type = "Maximum" if self.find_max else "Minimum"
            if self.metric_name:
                title = f'{self.metric_name.capitalize()} Evaluation - Finding {optimization_type} Value'
            else:
                title = f'{self.__class__.__name__} - Finding {optimization_type} Value'
        
        return super().plot(title=title, show_optimal=show_optimal, 
                           figsize=figsize, xlabel=xlabel, ylabel=ylabel)
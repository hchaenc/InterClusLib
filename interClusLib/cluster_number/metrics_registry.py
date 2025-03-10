class MetricRegistry:
    """
    Metric Registry class for managing and accessing different clustering evaluation metrics
    """
    def __init__(self):
        """Initialize empty registry"""
        self.metrics = {}
        self.properties = {}
    
    def register(self, name, function, maximize=False, description=None):
        """
        Register a new evaluation metric
        
        Parameters:
        name: str, metric name
        function: callable, function to calculate the metric
        maximize: bool, whether the metric should be maximized
        description: str, metric description
        
        Returns:
        function: registered function, allows use as a decorator
        """
        self.metrics[name] = function
        self.properties[name] = {
            'maximize': maximize,
            'description': description or name
        }
        return function
    
    def get_function(self, name):
        """
        Get the metric calculation function
        
        Parameters:
        name: str, metric name
        
        Returns:
        callable: metric calculation function
        
        Raises:
        ValueError: if the metric is not registered
        """
        if name not in self.metrics:
            raise ValueError(f"Unregistered evaluation metric: {name}")
        return self.metrics[name]
    
    def should_maximize(self, name):
        """
        Determine if the metric should be maximized
        
        Parameters:
        name: str, metric name
        
        Returns:
        bool: True if it should be maximized, False otherwise
        
        Raises:
        ValueError: if the metric is not registered
        """
        if name not in self.properties:
            raise ValueError(f"Unregistered evaluation metric: {name}")
        return self.properties[name]['maximize']
    
    def get_description(self, name):
        """
        Get the metric description
        
        Parameters:
        name: str, metric name
        
        Returns:
        str: metric description
        
        Raises:
        ValueError: if the metric is not registered
        """
        if name not in self.properties:
            raise ValueError(f"Unregistered evaluation metric: {name}")
        return self.properties[name]['description']
    
    def get_all_metrics(self):
        """
        Get all registered metric names
        
        Returns:
        list: all registered metric names
        """
        return list(self.metrics.keys())

# Create global registry instance
metric_registry = MetricRegistry()

# Try to register common metrics
try:
    # Import evaluation metric functions
    from interClusLib.evaluation import (
        distortion_score,
        silhouette_score,
        calinski_harabasz_index,
        davies_bouldin_index,
        dunn_index
    )
    
    # Register evaluation metrics
    metric_registry.register(
        'distortion', 
        distortion_score, 
        maximize=False, 
        description='Distortion Score (lower is better)'
    )
    
    metric_registry.register(
        'silhouette', 
        silhouette_score, 
        maximize=True, 
        description='Silhouette Score (higher is better)'
    )
    
    metric_registry.register(
        'calinski_harabasz', 
        calinski_harabasz_index, 
        maximize=True, 
        description='Calinski-Harabasz Index (higher is better)'
    )
    
    metric_registry.register(
        'davies_bouldin', 
        davies_bouldin_index, 
        maximize=False, 
        description='Davies-Bouldin Index (lower is better)'
    )
    
    metric_registry.register(
        'dunn', 
        dunn_index, 
        maximize=True, 
        description='Dunn Index (higher is better)'
    )
except ImportError as e:
    print(f"Error when registering built-in evaluation metrics: {e}")
    print("Manual registration of evaluation metrics is required to use related functions")
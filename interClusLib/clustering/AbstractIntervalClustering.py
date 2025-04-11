from abc import ABC, abstractmethod
import numpy as np
import inspect
from interClusLib.metric import SIMILARITY_FUNCTIONS, DISTANCE_FUNCTIONS

class AbstractIntervalClustering(ABC):
    """
    Abstract base class for interval clustering algorithms.
    
    An interval clustering algorithm groups interval data of shape (n_samples, n_dims, 2)
    where the last dimension represents the lower and upper bounds of each interval.
    
    Attributes
    ----------
    n_clusters : int
        Number of clusters to find.
    distance_func : str or callable
        Name of the distance/similarity function to use or a custom function.
    isSim : bool
        Indicates whether the metric is a similarity function (True) or distance function (False).
    labels_ : ndarray
        Cluster labels for the training data after fitting.
    centroids_ : ndarray
        Cluster centroids of shape (n_clusters, n_dims, 2) after fitting.
    train_data : ndarray
        The training data used to fit the model.
    """
    
    # Available distance and similarity functions
    distance_funcs = set(DISTANCE_FUNCTIONS.keys())
    similarity_funcs = set(SIMILARITY_FUNCTIONS.keys())
    
    def __init__(self, n_clusters=2, distance_func='euclidean', is_similarity=False, **kwargs):
        """
        Initialize interval clustering algorithm.
        
        Parameters
        ----------
        n_clusters : int, default=2
            Number of clusters to find.
        distance_func : str or callable, default='euclidean'
            Name of the distance/similarity function to use or a custom function.
            If a custom function is provided, it should take two interval arrays as input
            and return a float distance/similarity value.
        is_similarity : bool, default=False
            If providing a custom function, set to True if it's a similarity function,
            False if it's a distance function. Ignored if distance_func is a string.
        **kwargs : dict
            Additional parameters specific to the clustering algorithm.
        """
        self.n_clusters = n_clusters
        self.distance_func = distance_func
        
        # Check if distance_func is a callable (custom function) or a string (predefined function name)
        if callable(distance_func):
            self.isSim = is_similarity
            self.distance_function = distance_func
            # Validate function signature - should accept two arguments
            sig = inspect.signature(distance_func)
            if len(sig.parameters) != 2:
                raise ValueError(f"Custom distance function should accept exactly 2 arguments, but got {len(sig.parameters)}")
        else:
            # Handle predefined functions by name
            if self.distance_func in self.distance_funcs:
                self.isSim = False
                self.distance_function = DISTANCE_FUNCTIONS[distance_func]
            elif self.distance_func in self.similarity_funcs:
                self.isSim = True
                self.distance_function = SIMILARITY_FUNCTIONS[distance_func]
            else:
                valid_funcs = ", ".join(list(self.similarity_funcs) + list(self.distance_funcs))
                raise ValueError(f"Invalid distance function '{distance_func}'. Available options: {valid_funcs}")
        
        # Will be set during fitting
        self.labels_ = None
        self.centroids_ = None
        self.train_data = None
    
    @abstractmethod
    def fit(self, intervals):
        """
        Fit the clustering model to the interval data.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data to cluster with shape (n_samples, n_dims, 2)
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        pass
    
    def get_labels(self):
        """
        Get the cluster labels for the training data.
        
        Returns
        -------
        ndarray
            Cluster labels with shape (n_samples,)
            
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.labels_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.labels_
    
    @abstractmethod
    def compute_metrics_for_k_range(self, intervals, min_clusters=2, max_clusters=10, 
                                   metrics=['distortion', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'dunn'],
                                   **kwargs):
        """
        Compute evaluation metrics for a range of cluster numbers.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data with shape (n_samples, n_dims, 2)
        min_clusters : int, default=2
            Minimum number of clusters to evaluate
        max_clusters : int, default=10
            Maximum number of clusters to evaluate
        metrics : list of str, default=['distortion', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'dunn']
            Metrics to compute
        **kwargs : dict
            Additional parameters specific to the clustering algorithm
        
        Returns
        -------
        dict
            Dictionary where keys are metric names and values are dictionaries 
            mapping k values to metric results
        """
        pass
    
    @abstractmethod
    def cluster_and_return(self, data, k):
        """
        Run clustering algorithm on data and return labels and centroids.
        
        Parameters
        ----------
        data : ndarray
            Interval data with shape (n_samples, n_dims, 2)
        k : int
            Number of clusters
            
        Returns
        -------
        tuple
            (labels, centroids) - Cluster labels and centroids
        """
        pass
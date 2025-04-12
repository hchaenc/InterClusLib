import numpy as np
from warnings import warn
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram as scipy_dendrogram
from interClusLib.clustering.AbstractIntervalClustering import AbstractIntervalClustering

class IntervalAgglomerativeClustering(AbstractIntervalClustering):
    """
    An Agglomerative (Hierarchical) Clustering for interval data (n_dims, 2).
    Uses SciPy's linkage function to ensure correct hierarchical clustering structure.
    
    Interval data is structured as arrays with shape (n_samples, n_dims, 2),
    where the last dimension represents the lower and upper bounds of each interval.
    """
    # Mapping from our linkage names to SciPy's linkage method names
    _linkage_map = {
        'single': 'single',
        'complete': 'complete',
        'average': 'average',
        'ward': 'ward'
    }

    def __init__(self, n_clusters=2, linkage='average', distance_func='euclidean', is_similarity=None, **kwargs):
        """
        Initialize the interval clustering algorithm.
        
        Parameters:
        -----------
        n_clusters: int, default=2
            Number of clusters to find.
        linkage: str, default='average'
            Linkage criterion ('complete', 'average', 'single', 'ward').
        distance_func: str or callable, default='euclidean'
            Name of the distance/similarity function to use or a custom function.
            If using a predefined function, it can be from the distance_funcs or similarity_funcs.
        is_similarity: bool, default=None
            If providing a custom distance_func, set to True if it's a similarity function,
            False if it's a distance function. Ignored if distance_func is a string.
        **kwargs: dict
            Additional parameters to pass to the parent class.
        """
        if linkage not in self._linkage_map:
            raise ValueError(f"Unsupported linkage method: {linkage}. "
                           f"Supported methods are: {list(self._linkage_map.keys())}")
        
        # Call parent class constructor with appropriate parameters
        super().__init__(n_clusters=n_clusters, distance_func=distance_func, is_similarity=is_similarity, **kwargs)
        
        self.linkage = linkage
        
        # Additional attributes for hierarchical clustering
        self.n_samples_ = None
        self.linkage_matrix_ = None

    def compute_distance_matrix(self, intervals):
        """
        Compute the pairwise distance/similarity matrix for interval data.
        
        Parameters:
        -----------
        intervals: numpy.ndarray
            Interval data with shape (n_samples, n_dims, 2)
            
        Returns:
        --------
        numpy.ndarray
            Distance matrix with shape (n_samples, n_samples)
        """
        # Import necessary functions based on the metric type
        if self.isSim:
            # For similarity functions, compute similarity and convert to distance
            from interClusLib.metric import pairwise_similarity
            dist_matrix = pairwise_similarity(
                intervals, 
                metric=self.distance_function if callable(self.distance_func) else self.distance_func
            )
            dist_matrix = 1 - dist_matrix  # Convert similarity to distance
        else:
            # For distance functions, compute distance directly
            from interClusLib.metric import pairwise_distance
            dist_matrix = pairwise_distance(
                intervals, 
                metric=self.distance_function if callable(self.distance_func) else self.distance_func
            )
        
        return dist_matrix

    def _compute_centroids(self, intervals, labels, n_clusters):
        """
        Compute the centroids for each cluster.
        
        Parameters:
        -----------
        intervals: numpy.ndarray
            Interval data with shape (n_samples, n_dims, 2)
        labels: numpy.ndarray
            Cluster labels with shape (n_samples,)
        n_clusters: int
            Number of clusters
            
        Returns:
        --------
        numpy.ndarray
            Cluster centroids with shape (n_clusters, n_dims, 2)
        """
        centroids = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                cluster_data = intervals[cluster_indices]
                # Calculate the mean of intervals for each dimension
                center = np.mean(cluster_data, axis=0)
                centroids.append(center)
            else:
                # Handle the case when no points are assigned to this cluster
                print(f"Warning: Cluster {cluster_id} is empty.")
                # Create a zero array with matching shape as centroid
                center = np.zeros((intervals.shape[1], 2))
                centroids.append(center)
        
        return np.array(centroids)

    def fit(self, intervals):
        """
        Fit the hierarchical clustering model to the interval data.
        
        Parameters:
        -----------
        intervals: numpy.ndarray
            Interval data to cluster with shape (n_samples, n_dims, 2)
            
        Returns:
        --------
        self: object
            Fitted estimator.
        """
        # Store the input data
        self.train_data = intervals
        
        # Get number of samples
        n_samples = intervals.shape[0]
        self.n_samples_ = n_samples
        
        # Compute the distance matrix for the interval data
        dist_matrix = self.compute_distance_matrix(intervals)
        
        # Prepare condensed distance matrix for SciPy
        # SciPy expects a condensed distance matrix for linkage computation
        # (i.e., only the upper triangular portion without the diagonal)
        condensed_dist = []
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                condensed_dist.append(dist_matrix[i, j])
        
        # Use SciPy's linkage function to compute the hierarchical clustering
        # This ensures proper calculation of the linkage matrix
        scipy_method = self._linkage_map[self.linkage]
        
        # Special case for Ward linkage which requires squared Euclidean distances
        if self.linkage == 'ward' and self.distance_func != 'euclidean':
            warn("Ward linkage is designed to work with Euclidean distance. "
                "The results may not be optimal with other distance metrics.")
        
        # Compute the linkage matrix using SciPy
        self.linkage_matrix_ = linkage(
            condensed_dist, 
            method=scipy_method, 
            metric='precomputed'
        )
        
        # Use SciPy's fcluster to obtain cluster assignments
        # We use the 'maxclust' criterion to get a specific number of clusters
        cluster_assignments = fcluster(
            self.linkage_matrix_, 
            t=self.n_clusters, 
            criterion='maxclust'
        )
        
        # Convert to 0-based indexing (SciPy returns 1-based indices)
        self.labels_ = cluster_assignments - 1
        
        # Calculate and store cluster centroids for later use
        self.centroids_ = self._compute_centroids(intervals, self.labels_, self.n_clusters)
        
        return self

    def get_dendrogram_data(self):
        """
        Get the necessary data for plotting a dendrogram.
        
        Returns:
        --------
        dict
            Dictionary containing:
            - 'linkage_matrix': The linkage matrix for scipy dendrogram
            - 'labels': Cluster labels for the training data
            - 'n_leaves': Number of leaf nodes
            
        Raises:
        -------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.linkage_matrix_ is None:
            raise RuntimeError("Model not fitted yet.")
        
        return {
            'linkage_matrix': self.linkage_matrix_,
            'labels': self.labels_,
            'n_leaves': self.n_samples_
        }

    def compute_metrics_for_k_range(self, intervals, min_clusters=2, max_clusters=10,
                               metrics=['distortion', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'dunn'],
                               distance_func=None, linkage=None, **kwargs):
        """
        Compute evaluation metrics for a range of cluster numbers for hierarchical clustering.
        This is useful for determining the optimal number of clusters.

        Parameters:
        -----------
        intervals: numpy.ndarray
            Interval data with shape (n_samples, n_dims, 2)
        min_clusters: int, default=2
            Minimum number of clusters to evaluate
        max_clusters: int, default=10
            Maximum number of clusters to evaluate
        metrics: list of str, default=['distortion', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'dunn']
            Metrics to compute, can be any key from the EVALUATION dictionary
        distance_func: str or callable, default=None
            Distance function name or callable. If None, uses the current instance's distance function.
        linkage: str, default=None
            Linkage criterion. If None, uses the current instance's linkage.
        **kwargs: dict
            Additional parameters (not used but included for compatibility with ABC)

        Returns:
        --------
        dict
            Dictionary where keys are metric names and values are dictionaries
            mapping k values to metric results
        """
        from interClusLib.evaluation import EVALUATION
        # Import scipy's linkage function with a different name to avoid naming conflict
        from scipy.cluster.hierarchy import linkage as scipy_linkage_func

        # Check if requested metrics are valid
        for metric in metrics:
            if metric not in EVALUATION:
                raise ValueError(f"Unknown metric: {metric}. Available options: {list(EVALUATION.keys())}")

        # Use current instance parameters if not specified
        distance_func = distance_func or self.distance_func
        linkage_method = linkage or self.linkage

        # Initialize results dictionary
        results = {metric: {} for metric in metrics}

        # Check if distance_func is a callable
        distance_metric = distance_func
        if callable(distance_func):
            # For custom function, we'll use it directly with compute_distance_matrix
            is_similarity = kwargs.get('is_similarity', self.isSim if hasattr(self, 'isSim') else False)
        else:
            # For predefined function names
            if distance_func in self.distance_funcs:
                is_similarity = False
            elif distance_func in self.similarity_funcs:
                is_similarity = True
            else:
                valid_funcs = ", ".join(list(self.similarity_funcs) + list(self.distance_funcs))
                raise ValueError(f"Invalid distance function '{distance_func}'. Available options: {valid_funcs}")

        # Compute the distance matrix
        # Create a temporary instance to use compute_distance_matrix
        temp_instance = self.__class__(n_clusters=2, linkage=linkage_method, distance_func=distance_func)
        dist_matrix = temp_instance.compute_distance_matrix(intervals)

        # Prepare condensed distance matrix for SciPy
        n_samples = intervals.shape[0]
        condensed_dist = []
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                condensed_dist.append(dist_matrix[i, j])

        # Compute linkage matrix using SciPy
        scipy_method = self._linkage_map[linkage_method]
        linkage_matrix = scipy_linkage_func(condensed_dist, method=scipy_method, metric='precomputed')
        
        # Compute metrics for each k value
        for k in range(min_clusters, max_clusters + 1):
            try:
                # Cut the tree to get cluster assignments for k clusters
                labels = fcluster(linkage_matrix, t=k, criterion='maxclust') - 1
                
                # Compute centroids for the current clustering
                centroids = []
                for cluster_id in range(k):
                    cluster_indices = np.where(labels == cluster_id)[0]
                    if len(cluster_indices) > 0:
                        centroids.append(np.mean(intervals[cluster_indices], axis=0))
                    else:
                        centroids.append(np.zeros((intervals.shape[1], 2)))
                centroids = np.array(centroids)
                
                # Calculate all requested metrics
                for metric_name in metrics:
                    try:
                        metric_func = EVALUATION[metric_name]
                        metric_value = metric_func(
                            data=intervals,
                            labels=labels,
                            centers=centroids,
                            metric=distance_metric
                        )
                        results[metric_name][k] = metric_value
                    except Exception as e:
                        print(f"Error calculating {metric_name} for k={k}: {e}")
            except Exception as e:
                print(f"Error computing metrics for k={k}: {e}")
        
        return results
    
    def cluster_and_return(self, data, k):
        """
        Run hierarchical clustering on data and return labels and centroids.
        
        Parameters:
        -----------
        data : ndarray
            Interval data with shape (n_samples, n_dims, 2)
        k : int
            Number of clusters
            
        Returns:
        -------
        tuple
            (labels, centroids) - Cluster labels and centroids
        """
        model = self.__class__(
            n_clusters=k,
            linkage=self.linkage,
            distance_func=self.distance_func,
            is_similarity=self.isSim if hasattr(self, 'isSim') else None
        )
        model.fit(data)
        return model.labels_, model.centroids_
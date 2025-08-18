import numpy as np
from warnings import warn
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster, dendrogram as scipy_dendrogram
from scipy.spatial.distance import squareform
from interClusLib.clustering.AbstractIntervalClustering import AbstractIntervalClustering

class IntervalAgglomerativeClustering(AbstractIntervalClustering):
    """
    An optimized Agglomerative (Hierarchical) Clustering for interval data.
    
    This implementation features vectorized distance computations, efficient memory usage,
    and improved performance for large datasets while maintaining compatibility with 
    SciPy's hierarchical clustering functions.
    
    Interval data is structured as arrays with shape (n_samples, n_dims, 2),
    where the last dimension represents the lower and upper bounds of each interval.
    
    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters to find.
    linkage : str, default='average'
        Linkage criterion ('complete', 'average', 'single', 'ward').
    distance_func : str or callable, default='euclidean'
        Distance/similarity function name or custom function.
    is_similarity : bool, default=None
        Whether the custom function is a similarity (True) or distance (False) measure.
    compute_full_tree : bool, default=False
        Whether to compute the full tree for multiple cluster numbers.
    """
    
    # Mapping from our linkage names to SciPy's linkage method names
    _linkage_map = {
        'single': 'single',
        'complete': 'complete',
        'average': 'average',
        'ward': 'ward'
    }

    def __init__(self, n_clusters=2, linkage='average', distance_func='euclidean', 
                 is_similarity=None, compute_full_tree=False, **kwargs):
        """Initialize the interval clustering algorithm."""
        if linkage not in self._linkage_map:
            raise ValueError(f"Unsupported linkage method: {linkage}. "
                           f"Supported methods are: {list(self._linkage_map.keys())}")
        
        # Call parent class constructor with appropriate parameters
        super().__init__(n_clusters=n_clusters, distance_func=distance_func, 
                        is_similarity=is_similarity, **kwargs)
        
        self.linkage = linkage
        self.compute_full_tree = compute_full_tree
        
        # Additional attributes for hierarchical clustering
        self.n_samples_ = None
        self.linkage_matrix_ = None
        self._distance_matrix_cache = None

    def _compute_distance_matrix_vectorized(self, intervals):
        """
        Compute the pairwise distance/similarity matrix using vectorized operations.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data with shape (n_samples, n_dims, 2)
            
        Returns
        -------
        ndarray
            Distance matrix with shape (n_samples, n_samples)
        """
        n_samples = intervals.shape[0]
        
        try:
            # Try vectorized computation first
            dist_matrix = self.distance_function(intervals, intervals)
            
            # Handle different output shapes
            if dist_matrix.ndim == 1:
                # Convert to square matrix if needed
                dist_matrix = squareform(dist_matrix)
            elif dist_matrix.shape != (n_samples, n_samples):
                raise ValueError(f"Unexpected distance matrix shape: {dist_matrix.shape}")
            
            # Convert similarity to distance if needed
            if self.isSim:
                dist_matrix = 1.0 - dist_matrix
            
            # Ensure diagonal is zero and matrix is symmetric
            np.fill_diagonal(dist_matrix, 0.0)
            dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
            
        except Exception as e:
            print(f"Vectorized distance computation failed, using pairwise approach: {e}")
            dist_matrix = self._compute_distance_matrix_pairwise(intervals)
        
        return dist_matrix

    def _compute_distance_matrix_pairwise(self, intervals):
        """
        Fallback method for computing distance matrix using pairwise calculations.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data with shape (n_samples, n_dims, 2)
            
        Returns
        -------
        ndarray
            Distance matrix with shape (n_samples, n_samples)
        """
        n_samples = intervals.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float64)
        
        # Compute upper triangular part only (matrix is symmetric)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = self.distance_function(intervals[i], intervals[j])
                if self.isSim:
                    dist = 1.0 - dist  # Convert similarity to distance
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist  # Symmetric
        
        return dist_matrix

    def compute_distance_matrix(self, intervals):
        """
        Compute the pairwise distance/similarity matrix for interval data.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data with shape (n_samples, n_dims, 2)
            
        Returns
        -------
        ndarray
            Distance matrix with shape (n_samples, n_samples)
        """
        return self._compute_distance_matrix_vectorized(intervals)

    def _compute_centroids(self, intervals, labels, n_clusters):
        """
        Compute the centroids for each cluster using vectorized operations.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data with shape (n_samples, n_dims, 2)
        labels : ndarray
            Cluster labels with shape (n_samples,)
        n_clusters : int
            Number of clusters
            
        Returns
        -------
        ndarray
            Cluster centroids with shape (n_clusters, n_dims, 2)
        """
        n_dims = intervals.shape[1]
        centroids = np.zeros((n_clusters, n_dims, 2), dtype=np.float64)
        
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_data = intervals[mask]
            
            if len(cluster_data) > 0:
                # Vectorized mean computation
                centroids[cluster_id] = np.mean(cluster_data, axis=0)
            else:
                print(f"Warning: Cluster {cluster_id} is empty.")
                # Keep zero-initialized centroid
        
        return centroids

    def _prepare_condensed_matrix(self, dist_matrix):
        """
        Prepare condensed distance matrix for SciPy linkage function.
        
        Parameters
        ----------
        dist_matrix : ndarray
            Square distance matrix with shape (n_samples, n_samples)
            
        Returns
        -------
        ndarray
            Condensed distance matrix (upper triangular without diagonal)
        """
        n_samples = dist_matrix.shape[0]
        
        # Use numpy's advanced indexing for efficient extraction
        indices = np.triu_indices(n_samples, k=1)
        condensed_dist = dist_matrix[indices]
        
        return condensed_dist

    def fit(self, intervals):
        """
        Fit the hierarchical clustering model to the interval data.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data to cluster with shape (n_samples, n_dims, 2)
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Convert to numpy array and validate
        intervals = np.asarray(intervals, dtype=np.float64)
        
        if intervals.ndim != 3 or intervals.shape[2] != 2:
            raise ValueError(f"Expected intervals with shape (n_samples, n_dims, 2), got {intervals.shape}")
        
        # Store the input data
        self.train_data = intervals
        self.n_samples_ = intervals.shape[0]
        
        if self.n_samples_ < 2:
            raise ValueError("Need at least 2 samples for clustering")
        
        if self.n_clusters > self.n_samples_:
            raise ValueError(f"Number of clusters ({self.n_clusters}) cannot exceed number of samples ({self.n_samples_})")
        
        # Compute the distance matrix
        dist_matrix = self.compute_distance_matrix(intervals)
        
        # Cache distance matrix if computing full tree
        if self.compute_full_tree:
            self._distance_matrix_cache = dist_matrix
        
        # Prepare condensed distance matrix for SciPy
        condensed_dist = self._prepare_condensed_matrix(dist_matrix)
        
        # Special case for Ward linkage
        scipy_method = self._linkage_map[self.linkage]
        if self.linkage == 'ward' and self.distance_func != 'euclidean':
            warn("Ward linkage is designed to work with Euclidean distance. "
                "The results may not be optimal with other distance metrics.")
        
        # Compute the linkage matrix using SciPy
        self.linkage_matrix_ = scipy_linkage(
            condensed_dist, 
            method=scipy_method, 
            metric='precomputed'
        )
        
        # Get cluster assignments for the specified number of clusters
        cluster_assignments = fcluster(
            self.linkage_matrix_, 
            t=self.n_clusters, 
            criterion='maxclust'
        )
        
        # Convert to 0-based indexing (SciPy returns 1-based indices)
        self.labels_ = cluster_assignments - 1
        
        # Calculate and store cluster centroids
        self.centroids_ = self._compute_centroids(intervals, self.labels_, self.n_clusters)
        
        return self

    def predict(self, intervals, n_clusters=None):
        """
        Predict cluster labels for new interval data using the fitted tree.
        
        Note: This method assigns new points to the closest existing centroid.
        For true hierarchical prediction, the tree would need to be rebuilt.
        
        Parameters
        ----------
        intervals : ndarray
            New interval data with shape (n_samples, n_dims, 2)
        n_clusters : int, optional
            Number of clusters to use. If None, uses the fitted n_clusters.
            
        Returns
        -------
        ndarray
            Predicted cluster labels with shape (n_samples,)
        """
        if self.centroids_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        intervals = np.asarray(intervals, dtype=np.float64)
        n_samples = intervals.shape[0]
        
        # Use specified n_clusters or default to fitted value
        k = n_clusters or self.n_clusters
        
        # If different number of clusters requested, recompute from tree
        if k != self.n_clusters and self.linkage_matrix_ is not None:
            cluster_assignments = fcluster(
                self.linkage_matrix_, 
                t=k, 
                criterion='maxclust'
            )
            labels = cluster_assignments - 1
            centroids = self._compute_centroids(self.train_data, labels, k)
        else:
            centroids = self.centroids_
        
        # Assign new points to closest centroids
        predicted_labels = np.zeros(n_samples, dtype=np.int32)
        
        for i, sample in enumerate(intervals):
            best_cluster = 0
            best_distance = float('inf')
            
            for j, centroid in enumerate(centroids):
                dist = self.distance_function(sample, centroid)
                if self.isSim:
                    dist = 1.0 - dist  # Convert similarity to distance
                
                if dist < best_distance:
                    best_distance = dist
                    best_cluster = j
            
            predicted_labels[i] = best_cluster
        
        return predicted_labels

    def get_dendrogram_data(self):
        """
        Get the necessary data for plotting a dendrogram.
        
        Returns
        -------
        dict
            Dictionary containing dendrogram data
        """
        if self.linkage_matrix_ is None:
            raise RuntimeError("Model not fitted yet.")
        
        return {
            'linkage_matrix': self.linkage_matrix_,
            'labels': self.labels_,
            'n_leaves': self.n_samples_
        }

    def plot_dendrogram(self, **kwargs):
        """
        Plot the dendrogram of the hierarchical clustering.
        
        Parameters
        ----------
        **kwargs : dict
            Additional arguments to pass to scipy.cluster.hierarchy.dendrogram
            
        Returns
        -------
        dict
            Dendrogram data returned by scipy's dendrogram function
        """
        if self.linkage_matrix_ is None:
            raise RuntimeError("Model not fitted yet.")
        
        plt.figure(figsize=kwargs.pop('figsize', (10, 7)))
        dendrogram_data = scipy_dendrogram(self.linkage_matrix_, **kwargs)
        plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage} linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()
        
        return dendrogram_data

    def compute_metrics_for_k_range(self, intervals, min_clusters=2, max_clusters=10,
                               metrics=['distortion', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'dunn'],
                               distance_func=None, linkage=None, **kwargs):
        """
        Compute evaluation metrics for a range of cluster numbers efficiently.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data with shape (n_samples, n_dims, 2)
        min_clusters : int, default=2
            Minimum number of clusters to evaluate
        max_clusters : int, default=10
            Maximum number of clusters to evaluate
        metrics : list of str
            Metrics to compute
        distance_func : str or callable, optional
            Distance function to use
        linkage : str, optional
            Linkage criterion to use
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        dict
            Dictionary mapping metric names to k-value results
        """
        from interClusLib.evaluation import EVALUATION
        
        # Validate metrics
        for metric in metrics:
            if metric not in EVALUATION:
                raise ValueError(f"Unknown metric: {metric}. Available options: {list(EVALUATION.keys())}")

        # Use current instance parameters if not specified
        distance_func = distance_func or self.distance_func
        linkage_method = linkage or self.linkage

        # Initialize results dictionary
        results = {metric: {} for metric in metrics}

        # Check if we can reuse existing linkage matrix
        can_reuse_tree = (
            self.linkage_matrix_ is not None and 
            distance_func == self.distance_func and 
            linkage_method == self.linkage and
            self.compute_full_tree
        )

        if can_reuse_tree:
            print("Reusing existing linkage matrix for efficiency...")
            linkage_matrix = self.linkage_matrix_
            intervals_data = self.train_data
        else:
            print("Computing new linkage matrix...")
            # Create temporary instance for computation
            temp_instance = self.__class__(
                n_clusters=min_clusters, 
                linkage=linkage_method, 
                distance_func=distance_func,
                is_similarity=kwargs.get('is_similarity', self.isSim if hasattr(self, 'isSim') else None)
            )
            
            # Compute distance matrix and linkage
            dist_matrix = temp_instance.compute_distance_matrix(intervals)
            condensed_dist = temp_instance._prepare_condensed_matrix(dist_matrix)
            scipy_method = self._linkage_map[linkage_method]
            linkage_matrix = scipy_linkage(condensed_dist, method=scipy_method, metric='precomputed')
            intervals_data = intervals

        # Compute metrics for each k value
        for k in range(min_clusters, max_clusters + 1):
            print(f"Computing metrics for k={k}...")
            
            try:
                # Cut the tree to get cluster assignments for k clusters
                labels = fcluster(linkage_matrix, t=k, criterion='maxclust') - 1
                
                # Compute centroids efficiently
                centroids = self._compute_centroids(intervals_data, labels, k)
                
                # Calculate all requested metrics
                for metric_name in metrics:
                    try:
                        metric_func = EVALUATION[metric_name]
                        metric_value = metric_func(
                            data=intervals_data,
                            labels=labels,
                            centers=centroids,
                            metric=distance_func
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
        model = self.__class__(
            n_clusters=k,
            linkage=self.linkage,
            distance_func=self.distance_func,
            is_similarity=self.isSim if hasattr(self, 'isSim') else None
        )
        model.fit(data)
        return model.labels_, model.centroids_

    def get_cluster_tree_cuts(self, k_range):
        """
        Get cluster assignments for multiple k values efficiently.
        
        Parameters
        ----------
        k_range : list or range
            Range of k values to compute
            
        Returns
        -------
        dict
            Dictionary mapping k values to cluster labels
        """
        if self.linkage_matrix_ is None:
            raise RuntimeError("Model not fitted yet.")
        
        results = {}
        for k in k_range:
            if k < 1 or k > self.n_samples_:
                print(f"Warning: k={k} is outside valid range [1, {self.n_samples_}]")
                continue
                
            cluster_assignments = fcluster(
                self.linkage_matrix_, 
                t=k, 
                criterion='maxclust'
            )
            results[k] = cluster_assignments - 1  # Convert to 0-based indexing
        
        return results

    def fit_predict(self, intervals):
        """
        Fit the model and return cluster labels.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data with shape (n_samples, n_dims, 2)
            
        Returns
        -------
        ndarray
            Cluster labels with shape (n_samples,)
        """
        return self.fit(intervals).labels_
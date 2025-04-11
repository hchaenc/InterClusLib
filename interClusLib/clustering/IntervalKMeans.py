import numpy as np
from numpy.random import RandomState
from warnings import warn
from interClusLib.clustering.AbstractIntervalClustering import AbstractIntervalClustering

class IntervalKMeans(AbstractIntervalClustering):
    """
    A K-Means clustering algorithm for interval data.
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to find.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    distance_func : str, default='euclidean'
        Distance function name.
    random_state : int or RandomState, default=42
        Random seed or RandomState for initialization.
    """

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, distance_func='euclidean', random_state=42):
        # Call parent class constructor
        super().__init__(n_clusters=n_clusters, distance_func=distance_func)
        
        self.max_iter = max_iter
        self.tol = tol
        
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            try:
                self.random_state = np.random.RandomState(random_state)
            except:
                print(f"Warning: Could not use random_state={random_state}, using default seed 42 instead")
                self.random_state = np.random.RandomState(42)

    def _init_centroids(self, intervals):
        """
        Initialize cluster centroids by randomly picking samples from 'intervals'.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data with shape (n_samples, n_dims, 2)
            
        Returns
        -------
        ndarray
            Initial centroids with shape (n_clusters, n_dims, 2)
        """
        n_samples = intervals.shape[0]
        # randomly choose k distinct samples as initial centroids
        indices = self.random_state.choice(n_samples, self.n_clusters, replace=False)
        centroids = intervals[indices].copy()
        return centroids

    def _compute_centroid(self, intervals_in_cluster):
        """
        Compute the centroid of intervals in one cluster.
        
        Parameters
        ----------
        intervals_in_cluster : ndarray
            Interval data with shape (k, n_dims, 2)
            
        Returns
        -------
        ndarray
            Centroid with shape (n_dims, 2)
        """
        # mean of lower bounds, mean of upper bounds dimension-wise
        return np.mean(intervals_in_cluster, axis=0)
    
    def _assign_clusters(self, intervals, centroids):
        """
        Assign each sample in 'intervals' to the nearest centroid.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data with shape (n_samples, n_dims, 2)
        centroids : ndarray
            Centroid data with shape (n_clusters, n_dims, 2)
            
        Returns
        -------
        ndarray
            Cluster labels with shape (n_samples,)
        """
        n_samples = intervals.shape[0]
        labels = np.zeros(n_samples, dtype=np.int32)

        for i in range(n_samples):
            # compute distance to each centroid
            dists = [self.distance_function(intervals[i], c) for c in centroids]
            if self.isSim:
                labels[i] = np.argmax(dists)  # For similarity: choose highest value
            else:
                labels[i] = np.argmin(dists)  # For distance: choose lowest value
        return labels
    
    def fit(self, intervals):
        """
        Fit the K-Means model to the interval data.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data to cluster with shape (n_samples, n_dims, 2)
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # 1. Initialize centroids
        centroids = self._init_centroids(intervals)

        for iteration in range(self.max_iter):
            # 2. Assign clusters
            labels = self._assign_clusters(intervals, centroids)

            # 3. Compute new centroids
            new_centroids = []
            for k in range(self.n_clusters):
                cluster_points = intervals[labels == k]
                if len(cluster_points) > 0:
                    centroid_k = self._compute_centroid(cluster_points)
                else:
                    # if no points assigned, keep previous centroid
                    centroid_k = centroids[k]
                new_centroids.append(centroid_k)
            new_centroids = np.array(new_centroids)

            # 4. Check for convergence (centroid shift)
            shift = np.sum((centroids - new_centroids)**2)
            centroids = new_centroids
            if shift < self.tol:
                break

        # save final centroids and labels
        self.train_data = intervals
        self.centroids_ = centroids
        self.labels_ = labels
        
        return self

    def compute_metrics_for_k_range(self, intervals, min_clusters=2, max_clusters=10, 
                               metrics=['distortion', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'dunn'],
                               distance_func=None, max_iter=None, tol=None, random_state=None, n_init=1):
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
        metrics : list, default=['distortion', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'dunn']
            Metrics to compute
        distance_func : str, default=None
            Distance function name. If None, uses the current instance's distance function.
        max_iter : int, default=None
            Maximum number of iterations. If None, uses the current instance's value.
        tol : float, default=None
            Convergence tolerance. If None, uses the current instance's value.
        random_state : int, default=None
            Random seed. If None, uses the current instance's value.
        n_init : int, default=1
            Number of times to run the algorithm with different centroid seeds.
        
        Returns
        -------
        dict
            Dictionary where keys are metric names and values are dictionaries 
            mapping k values to metric results
        """
        from interClusLib.evaluation import EVALUATION
        
        # Check if requested metrics are valid
        for metric in metrics:
            if metric not in EVALUATION:
                raise ValueError(f"Unknown metric: {metric}. Available options: {list(EVALUATION.keys())}")
        
        # Use current instance parameters if not specified
        distance_func = distance_func or self.distance_func
        max_iter = max_iter or self.max_iter
        tol = tol or self.tol
        random_state = random_state if random_state is not None else (
            self.random_state.randint(0, 10000) if isinstance(self.random_state, np.random.RandomState) 
            else self.random_state
        )
        
        # Initialize results dictionary
        results = {metric: {} for metric in metrics}
        
        # Compute metrics for each k value
        for k in range(min_clusters, max_clusters + 1):
            best_inertia = float('inf')
            best_model = None
            
            # Run multiple initializations
            for init in range(n_init):
                try:
                    model = self.__class__(
                        n_clusters=k,
                        max_iter=max_iter,
                        tol=tol,
                        distance_func=distance_func,
                        random_state=random_state + init if random_state is not None else None
                    )
                    model.fit(intervals)
                    
                    # Calculate inertia for best model selection
                    inertia = 0
                    for i, sample in enumerate(intervals):
                        cluster_idx = model.labels_[i]
                        centroid = model.centroids_[cluster_idx]
                        dist = model.distance_function(sample, centroid)
                        # Convert similarity to distance if needed
                        if model.isSim:
                            dist = 1 - dist
                        inertia += dist ** 2
                    
                    # Keep the best model based on inertia
                    if inertia < best_inertia:
                        best_inertia = inertia
                        best_model = model
                except Exception as e:
                    print(f"Error fitting model with k={k}, initialization {init}: {e}")
            
            if best_model is None:
                print(f"Failed to fit model for k={k}, skipping")
                continue
                
            # Calculate all requested metrics
            for metric in metrics:
                try:
                    metric_func = EVALUATION[metric]
                    metric_value = metric_func(
                        data=intervals,
                        labels=best_model.labels_,
                        centers=best_model.centroids_,
                        metric=distance_func
                    )
                    results[metric][k] = metric_value
                except Exception as e:
                    print(f"Error calculating {metric} for k={k}: {e}")
        
        return results

    def cluster_and_return(self, data, k):
        """
        Run K-Means clustering algorithm on data and return labels and centroids.
        
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
            max_iter=self.max_iter,
            tol=self.tol,
            distance_func=self.distance_func,
            random_state=self.random_state
        )
        model.fit(data)
        return model.labels_, model.centroids_
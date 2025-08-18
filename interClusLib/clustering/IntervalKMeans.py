import numpy as np
from numpy.random import RandomState
from warnings import warn
from interClusLib.clustering.AbstractIntervalClustering import AbstractIntervalClustering

class IntervalKMeans(AbstractIntervalClustering):
    """
    An optimized K-Means clustering algorithm for interval data.
    
    This implementation features vectorized operations, multiple initialization 
    strategies, and improved performance for large datasets.
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to find.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    distance_func : str or callable, default='euclidean'
        Distance function name or custom function.
    init : str, default='k-means++'
        Initialization method ('random' or 'k-means++').
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different centroid seeds.
    random_state : int or RandomState, default=42
        Random seed or RandomState for initialization.
    """

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, distance_func='euclidean', 
                 init='k-means++', n_init=10, random_state=42):
        # Call parent class constructor
        super().__init__(n_clusters=n_clusters, distance_func=distance_func)
        
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.n_init = n_init
        
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            try:
                self.random_state = np.random.RandomState(random_state)
            except:
                print(f"Warning: Could not use random_state={random_state}, using default seed 42 instead")
                self.random_state = np.random.RandomState(42)

    def _init_centroids_random(self, intervals):
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
        # Randomly choose k distinct samples as initial centroids
        indices = self.random_state.choice(n_samples, self.n_clusters, replace=False)
        centroids = intervals[indices].copy()
        return centroids

    def _init_centroids_plus_plus(self, intervals):
        """
        Initialize centroids using K-means++ algorithm for better initial placement.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data with shape (n_samples, n_dims, 2)
            
        Returns
        -------
        ndarray
            Initial centroids with shape (n_clusters, n_dims, 2)
        """
        n_samples, n_dims, _ = intervals.shape
        centroids = np.zeros((self.n_clusters, n_dims, 2))
        
        # Choose first centroid randomly
        centroids[0] = intervals[self.random_state.randint(n_samples)]
        
        # Choose remaining centroids
        for k in range(1, self.n_clusters):
            # Compute distances to nearest centroid for each point
            distances = np.full(n_samples, np.inf)
            
            for i in range(n_samples):
                min_dist = np.inf
                for j in range(k):
                    dist = self.distance_function(intervals[i], centroids[j])
                    if self.isSim:
                        dist = 1 - dist  # Convert similarity to distance
                    min_dist = min(min_dist, dist)
                distances[i] = min_dist
            
            # Choose next centroid with probability proportional to squared distance
            distances_squared = distances ** 2
            probabilities = distances_squared / np.sum(distances_squared)
            cumulative_probs = np.cumsum(probabilities)
            r = self.random_state.rand()
            next_centroid_idx = np.searchsorted(cumulative_probs, r)
            centroids[k] = intervals[next_centroid_idx]
        
        return centroids

    def _init_centroids(self, intervals):
        """
        Initialize cluster centroids using the specified method.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data with shape (n_samples, n_dims, 2)
            
        Returns
        -------
        ndarray
            Initial centroids with shape (n_clusters, n_dims, 2)
        """
        if self.init == 'k-means++':
            return self._init_centroids_plus_plus(intervals)
        elif self.init == 'random':
            return self._init_centroids_random(intervals)
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")

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
        if len(intervals_in_cluster) == 0:
            # Return zero centroid if cluster is empty
            return np.zeros((intervals_in_cluster.shape[1], intervals_in_cluster.shape[2]))
        # Mean of lower bounds, mean of upper bounds dimension-wise
        return np.mean(intervals_in_cluster, axis=0)
    
    def _assign_clusters(self, intervals, centroids):
        """
        Vectorized assignment of each sample to the nearest centroid.
        
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
        # Use vectorized distance computation if available
        try:
            # Compute distance matrix: (n_samples, n_clusters)
            distance_matrix = self.distance_function(intervals, centroids)
            
            # Handle the output format based on the distance function
            if distance_matrix.ndim == 1:
                # Single sample case, reshape to matrix
                distance_matrix = distance_matrix.reshape(1, -1)
            elif distance_matrix.ndim == 0:
                # Scalar case, this shouldn't happen but handle it
                distance_matrix = np.array([[distance_matrix]])
            
            # Assign to closest centroid
            if self.isSim:
                labels = np.argmax(distance_matrix, axis=1)  # For similarity: choose highest value
            else:
                labels = np.argmin(distance_matrix, axis=1)  # For distance: choose lowest value
                
        except Exception as e:
            # Fallback to loop-based assignment if vectorization fails
            print(f"Vectorized assignment failed, falling back to loop: {e}")
            labels = self._assign_clusters_loop(intervals, centroids)
        
        return labels.astype(np.int32)
    
    def _assign_clusters_loop(self, intervals, centroids):
        """
        Loop-based assignment (fallback method).
        
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
            # Compute distance to each centroid
            dists = []
            for c in centroids:
                dist = self.distance_function(intervals[i], c)
                dists.append(dist)
            
            if self.isSim:
                labels[i] = np.argmax(dists)  # For similarity: choose highest value
            else:
                labels[i] = np.argmin(dists)  # For distance: choose lowest value
        
        return labels
    
    def _compute_inertia(self, intervals, labels, centroids):
        """
        Compute the within-cluster sum of squares (inertia).
        
        Parameters
        ----------
        intervals : ndarray
            Interval data with shape (n_samples, n_dims, 2)
        labels : ndarray
            Cluster labels with shape (n_samples,)
        centroids : ndarray
            Centroid data with shape (n_clusters, n_dims, 2)
            
        Returns
        -------
        float
            Inertia value
        """
        inertia = 0.0
        for i, sample in enumerate(intervals):
            cluster_idx = labels[i]
            centroid = centroids[cluster_idx]
            dist = self.distance_function(sample, centroid)
            # Convert similarity to distance if needed
            if self.isSim:
                dist = 1 - dist
            inertia += dist ** 2
        return inertia
    
    def _fit_single(self, intervals):
        """
        Perform a single run of K-means clustering.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data with shape (n_samples, n_dims, 2)
            
        Returns
        -------
        tuple
            (labels, centroids, inertia, n_iter) - Results of single run
        """
        # 1. Initialize centroids
        centroids = self._init_centroids(intervals)
        prev_labels = None

        for iteration in range(self.max_iter):
            # 2. Assign clusters using vectorized method
            labels = self._assign_clusters(intervals, centroids)

            # 3. Check for convergence (no change in labels)
            if prev_labels is not None and np.array_equal(labels, prev_labels):
                break
            prev_labels = labels.copy()

            # 4. Compute new centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                cluster_points = intervals[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = self._compute_centroid(cluster_points)
                else:
                    # If no points assigned, keep previous centroid
                    new_centroids[k] = centroids[k]

            # 5. Check for convergence (centroid shift)
            centroid_shift = np.sum((centroids - new_centroids)**2)
            centroids = new_centroids
            if centroid_shift < self.tol:
                break

        # Compute final inertia
        inertia = self._compute_inertia(intervals, labels, centroids)
        
        return labels, centroids, inertia, iteration + 1
    
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
        intervals = np.asarray(intervals, dtype=np.float64)
        
        if intervals.shape[0] < self.n_clusters:
            raise ValueError(f"Number of samples ({intervals.shape[0]}) must be >= n_clusters ({self.n_clusters})")
        
        best_inertia = np.inf
        best_labels = None
        best_centroids = None
        best_n_iter = 0
        
        # Run multiple initializations and keep the best result
        for init_run in range(self.n_init):
            try:
                # Use different random seed for each initialization
                original_state = self.random_state.get_state()
                self.random_state.seed(self.random_state.randint(0, 2**31) + init_run)
                
                labels, centroids, inertia, n_iter = self._fit_single(intervals)
                
                # Restore random state
                self.random_state.set_state(original_state)
                
                if inertia < best_inertia:
                    best_inertia = inertia
                    best_labels = labels
                    best_centroids = centroids
                    best_n_iter = n_iter
                    
            except Exception as e:
                print(f"Warning: Initialization {init_run} failed: {e}")
                continue

        if best_labels is None:
            raise RuntimeError("All initializations failed")

        # Save final results
        self.train_data = intervals
        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        return self

    def predict(self, intervals):
        """
        Predict cluster labels for new interval data.
        
        Parameters
        ----------
        intervals : ndarray
            Interval data with shape (n_samples, n_dims, 2)
            
        Returns
        -------
        ndarray
            Predicted cluster labels with shape (n_samples,)
        """
        if self.centroids_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        intervals = np.asarray(intervals, dtype=np.float64)
        return self._assign_clusters(intervals, self.centroids_)

    def fit_predict(self, intervals):
        """
        Fit the model and predict cluster labels.
        
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

    def compute_metrics_for_k_range(self, intervals, min_clusters=2, max_clusters=10, 
                               metrics=['distortion', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'dunn'],
                               distance_func=None, max_iter=None, tol=None, random_state=None, n_init=None):
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
        n_init : int, default=None
            Number of initializations. If None, uses the current instance's value.
        
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
        n_init = n_init or self.n_init
        random_state = random_state if random_state is not None else (
            self.random_state.randint(0, 10000) if isinstance(self.random_state, np.random.RandomState) 
            else self.random_state
        )
        
        # Initialize results dictionary
        results = {metric: {} for metric in metrics}
        
        # Compute metrics for each k value
        for k in range(min_clusters, max_clusters + 1):
            print(f"Computing metrics for k={k}...")
            
            try:
                model = self.__class__(
                    n_clusters=k,
                    max_iter=max_iter,
                    tol=tol,
                    distance_func=distance_func,
                    init=self.init,
                    n_init=n_init,
                    random_state=random_state
                )
                model.fit(intervals)
                
                # Calculate all requested metrics
                for metric in metrics:
                    try:
                        metric_func = EVALUATION[metric]
                        metric_value = metric_func(
                            data=intervals,
                            labels=model.labels_,
                            centers=model.centroids_,
                            metric=distance_func
                        )
                        results[metric][k] = metric_value
                    except Exception as e:
                        print(f"Error calculating {metric} for k={k}: {e}")
                        
            except Exception as e:
                print(f"Error fitting model for k={k}: {e}")
        
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
            init=self.init,
            n_init=self.n_init,
            random_state=self.random_state
        )
        model.fit(data)
        return model.labels_, model.centroids_
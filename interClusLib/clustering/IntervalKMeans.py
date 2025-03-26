import pandas as pd
import numpy as np
from numpy.random import RandomState
from warnings import warn
from interClusLib.metric import SIMILARITY_FUNCTIONS, DISTANCE_FUNCTIONS

class IntervalKMeans:
    """
    A custom K-Means clustering for interval data
    """

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, distance_func='euclidean', random_state=42):    
        """
        :param n_clusters: number of clusters
        :param max_iter: maximum number of iterations
        :param tol: tolerance for convergence
        :param distance_func: distance function name or callable
        :param random_state: random seed
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.train_data = None
        self.isSim = None
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            try:
                self.random_state = np.random.RandomState(random_state)
            except:
                # 如果转换失败，使用默认种子
                print(f"Warning: Could not use random_state={random_state}, using default seed 42 instead")
                self.random_state = np.random.RandomState(42)
        
        # 保存原始的distance_func名称，用于创建新实例时使用
        self.distance_func_name = distance_func if isinstance(distance_func, str) else 'custom'

        if distance_func in SIMILARITY_FUNCTIONS:
            self.distance_function = SIMILARITY_FUNCTIONS[distance_func]
            self.isSim = True
        elif distance_func in DISTANCE_FUNCTIONS:
            self.distance_function = DISTANCE_FUNCTIONS[distance_func]
            self.isSim = False
        else:
            valid_funcs = ", ".join(list(SIMILARITY_FUNCTIONS.keys()) + list(DISTANCE_FUNCTIONS.keys()))
            raise ValueError(f"Invalid distance function '{distance_func}'. Available options: {valid_funcs}")

    def _init_centroids(self, intervals):
        """
        Initialize cluster centroids by randomly picking samples from 'intervals'.
        intervals: shape (n_samples, n_dims, 2)
        """
        n_samples = intervals.shape[0]
        # randomly choose k distinct samples as initial centroids
        indices = self.random_state.choice(n_samples, self.n_clusters, replace=False)
        centroids = intervals[indices].copy()
        return centroids

    def _compute_centroid(self, intervals_in_cluster):
        """
        Compute the centroid of intervals in one cluster.
        intervals_in_cluster: shape (k, n_dims, 2)
        """
        # mean of lower bounds, mean of upper bounds dimension-wise
        return np.mean(intervals_in_cluster, axis=0)
    
    def _assign_clusters(self, intervals, centroids):
        """
        Assign each sample in 'intervals' to the nearest centroid using 'distance_func'.
        """
        n_samples = intervals.shape[0]
        labels = np.zeros(n_samples, dtype=np.int32)

        for i in range(n_samples):
            # compute distance to each centroid
            dists = [self.distance_function(intervals[i], c) for c in centroids]
            if self.isSim:
                labels[i] = np.argmax(dists)  # 相似性：选择值最大的 centroid
            else:
                labels[i] = np.argmin(dists)
        return labels
    
    def fit(self, intervals):
        """
        intervals: shape (n_samples, n_dims, 2)
        distance_func: function that takes (interval_a, interval_b) and returns a scalar distance
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
                    # if no points assigned, re-initialize or handle it in some way
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

    def get_labels(self):
        if self.labels_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.labels_

    def compute_metrics_for_k_range(self, intervals, min_clusters=2, max_clusters=10, 
                               metrics=['distortion', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'dunn'], distance_func=None, 
                               max_iter=None, tol=None, random_state=None, 
                               n_init=1):
        """
        Compute evaluation metrics for a range of cluster numbers.
        
        Parameters:
        -----------
        intervals : array-like
            Interval data with shape (n_samples, n_dims, 2)
        min_clusters : int, default=2
            Minimum number of clusters to evaluate
        max_clusters : int, default=10
            Maximum number of clusters to evaluate
        metrics : list of str, default=['distortion']
            Metrics to compute, can be any key from the EVALUATION dictionary
        distance_func : str or callable, default=None
            Distance function name or callable. If None, uses the current instance's distance function.
        max_iter : int, default=None
            Maximum number of iterations. If None, uses the current instance's value.
        tol : float, default=None
            Convergence tolerance. If None, uses the current instance's value.
        random_state : int, default=None
            Random seed. If None, uses the current instance's value.
        n_init : int, default=1
            Number of times to run the algorithm with different centroid seeds.
        
        Returns:
        --------
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
        distance_func = distance_func or self.distance_func_name
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
        对数据运行区间聚类并返回标签和中心点
        
        Parameters:
        -----------
        data : array-like
            形状为(n_samples, n_dims, 2)的区间数据
        k : int, optional
            聚类数量，如果为None则使用初始化时设置的n_clusters
            
        Returns:
        --------
        tuple
            (labels, centroids) - 聚类标签和中心点
        """
        model = IntervalKMeans(
            n_clusters=k,
            max_iter=self.max_iter,
            tol=self.tol,
            distance_func=self.distance_func_name,
            random_state=self.random_state
        )
        model.fit(data)
        return model.labels_, model.centroids_
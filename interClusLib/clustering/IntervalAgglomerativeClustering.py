import pandas as pd
import numpy as np
from numpy.random import RandomState
from warnings import warn
from sklearn.cluster import AgglomerativeClustering
from interClusLib.metric import *
import os

class IntervalAgglomerativeClustering:
    """
    An Agglomerative (Hierarchical) Clustering for interval data (n_dims, 2).
    Uses a precomputed distance matrix from a custom distance function.
    
    Interval data is structured as arrays with shape (n_samples, n_dims, 2),
    where the last dimension represents the lower and upper bounds of each interval.
    """
    # Available distance metrics for interval data
    distance_funcs = {"hausdorff", "euclidean", "manhattan"}
    # Available similarity metrics for interval data
    similarity_funcs = {"jaccard", "dice", "bidirectional_min","bidirectional_prod", "marginal"}

    def __init__(self, n_clusters=2, linkage='average', distance_func = 'euclidean'):
        """
        Initialize the interval clustering algorithm.
        
        Parameters:
        -----------
        n_clusters: int, default=2
            Number of clusters to find.
        linkage: str, default='average'
            Linkage criterion ('complete', 'average', 'single').
        distance_func: str, default='euclidean'
            Name of the distance/similarity function to use.
            Can be one of the distance_funcs or similarity_funcs.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage

        self.model_ = None
        self.labels_ = None
        self.distance_func = distance_func

        # Determine if the metric is a similarity or distance function
        if self.distance_func in self.distance_funcs:
            self.isSim = False
        elif self.distance_func in self.similarity_funcs:
            self.isSim = True
        else:
            raise ValueError(f"Unsupported metric: {self.distance_func}")
    
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
        if self.isSim:
            # For similarity functions, compute similarity and convert to distance
            dist_matrix = pairwise_similarity(
                intervals, 
                metric=self.distance_func
            )
            dist_matrix = 1 - dist_matrix  # Convert similarity to distance
        else:
            # For distance functions, compute distance directly
            dist_matrix = pairwise_distance(
                intervals, 
                metric=self.distance_func
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
        """
        # Compute the distance matrix for the interval data
        dist_matrix = self.compute_distance_matrix(intervals)

        # Initialize the scikit-learn AgglomerativeClustering with precomputed distances
        self.model_ = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric='precomputed', 
            linkage=self.linkage,
            compute_distances=True
        )
        self.model_.fit(dist_matrix)

        # Store the cluster labels and input data
        self.labels_ = self.model_.labels_
        self.train_data = intervals
        
        # Calculate and store cluster centroids for later use
        self.centroids_ = self._compute_centroids(intervals, self.labels_, self.n_clusters)

    def get_labels(self):
        """
        Get the cluster labels for the training data.
        
        Returns:
        --------
        numpy.ndarray
            Cluster labels with shape (n_samples,)
            
        Raises:
        -------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.labels_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.labels_
    
    def predict(self, intervals_new, method='knn', k=5):
        """
        Predict cluster labels for new interval data.
        
        Parameters:
        -----------
        intervals_new: numpy.ndarray
            New interval data with shape (n_samples, n_dims, 2)
        method: str, default='knn'
            Prediction method, either 'knn' or 'center'
        k: int, default=5
            Number of neighbors to consider in the knn method
            
        Returns:
        --------
        numpy.ndarray
            Predicted cluster labels with shape (n_samples,)
            
        Raises:
        -------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If an unsupported prediction method is specified.
        """
        if not hasattr(self, 'train_data') or self.labels_ is None:
            raise RuntimeError("Model not fitted yet.")
        
        if method == 'knn':
            return self._predict_knn(intervals_new, k)
        elif method == 'center':
            return self._predict_centers(intervals_new)
        else:
            raise ValueError(f"Unsupported prediction method: {method}")
    
    def _predict_knn(self, intervals_new, k):
        """
        Predict cluster labels using the k-nearest neighbors approach.
        
        Parameters:
        -----------
        intervals_new: numpy.ndarray
            New interval data with shape (n_samples, n_dims, 2)
        k: int
            Number of neighbors to consider
            
        Returns:
        --------
        numpy.ndarray
            Predicted cluster labels with shape (n_samples,)
        """
        # Calculate distances between new data and training data
        if self.isSim:
            cross_dist = cross_similarity(intervals_new, self.train_data, metric=self.distance_func)
            cross_dist = 1 - cross_dist  # Convert similarity to distance
        else:
            cross_dist = cross_distance(intervals_new, self.train_data, metric=self.distance_func)
        
        # For each new sample, find k nearest training samples
        predictions = []
        for i in range(len(intervals_new)):
            # Get indices of k nearest neighbors
            nearest_indices = np.argsort(cross_dist[i])[:k]
            nearest_labels = [self.labels_[idx] for idx in nearest_indices]
            
            # Vote to select the most common label
            from collections import Counter
            most_common = Counter(nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)
    
    def _predict_centers(self, intervals_new):
        """
        Predict cluster labels by finding the closest cluster center.
        
        Parameters:
        -----------
        intervals_new: numpy.ndarray
            New interval data with shape (n_samples, n_dims, 2)
            
        Returns:
        --------
        numpy.ndarray
            Predicted cluster labels with shape (n_samples,)
        """
        # Use the pre-computed centroids from the fit method
        centers = self.centroids_  # Shape: (n_clusters, n_dims, 2)
        
        # Calculate distances to each center
        predictions = []
        for interval in intervals_new:
            distances = []
            for center in centers:
                # Use the original distance calculation method
                combined = np.stack([interval, center])
                dist_matrix = self.compute_distance_matrix(combined)
                dist = dist_matrix[0, 1]  # Get the cross-distance
                
                distances.append(dist)
            
            # Assign to the closest center
            closest_center = np.argmin(distances)
            predictions.append(closest_center)
        
        return np.array(predictions)

    def compute_metrics_for_k_range(self, intervals, min_clusters=2, max_clusters=10, 
                       metrics=['distortion'], distance_func=None, 
                       linkage=None):
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
        metrics: list of str, default=['distortion']
            Metrics to compute, can be any key from the EVALUATION dictionary
        distance_func: str, default=None
            Distance function name. If None, uses the current instance's distance function.
        linkage: str, default=None
            Linkage criterion. If None, uses the current instance's linkage.
        
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
        distance_func = distance_func or self.distance_func
        linkage = linkage or self.linkage
        
        # Initialize results dictionary
        results = {metric: {} for metric in metrics}
        
        # Compute metrics for each k value
        for k in range(min_clusters, max_clusters + 1):
            try:
                # Create a new model instance with k clusters
                model = self.__class__(
                    n_clusters=k,
                    linkage=linkage,
                    distance_func=distance_func
                )
                model.fit(intervals)
                
                # Calculate all requested metrics
                for metric in metrics:
                    try:
                        metric_func = EVALUATION[metric]
                        
                        # Use the pre-computed centroids from the model
                        centers = model.centroids_
                        
                        # Compute the metric value
                        metric_value = metric_func(
                            data=intervals,
                            labels=model.labels_,
                            centers=centers,
                            metric=distance_func
                        )
                        results[metric][k] = metric_value
                    except Exception as e:
                        print(f"Error calculating {metric} for k={k}: {e}")
            except Exception as e:
                print(f"Error fitting model with k={k}: {e}")
        
        return results
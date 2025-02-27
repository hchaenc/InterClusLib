import pandas as pd
import numpy as np
from numpy.random import RandomState
from warnings import warn
from interClusLib.metric import MULTI_SIMILARITY_FUNCTIONS, MULTI_DISTANCE_FUNCTIONS
import os

class IntervalKMeans:
    """
    A custom K-Means clustering for interval data
    """

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, distance_func= 'euclidean',random_state=42):
        """
        :param n_clusters: number of clusters
        :param max_iter: maximum number of iterations
        :param tol: tolerance for convergence
        :param random_state: random seed
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = np.random.RandomState(random_state)
        self.train_data = None

        if distance_func in MULTI_SIMILARITY_FUNCTIONS:
            self.distance_function = MULTI_SIMILARITY_FUNCTIONS[distance_func]
            self.isSim = True
        elif distance_func in MULTI_DISTANCE_FUNCTIONS:
            self.distance_function = MULTI_DISTANCE_FUNCTIONS[distance_func]
            self.isSim = False
        else:
            valid_funcs = ", ".join(list(MULTI_SIMILARITY_FUNCTIONS.keys()) + list(MULTI_DISTANCE_FUNCTIONS.keys()))
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
        self.train_data  = intervals
        self.centroids_ = centroids
        self.labels_ = labels

    def predict(self, intervals):
        """
        Assign new data points to the closest cluster.
        """
        predictions = self._assign_clusters(intervals, self.centroids_)
        return predictions
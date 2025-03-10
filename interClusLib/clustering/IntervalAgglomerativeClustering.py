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
    """
    distance_funcs = {"hausdorff", "euclidean", "manhattan"}
    similarity_funcs = {"jaccard", "dice", "bidirectional_min","bidirectional_prod", "marginal"}

    def __init__(self, n_clusters=2, linkage='average', distance_func = 'euclidean'):
        """
        :param n_clusters: int, number of clusters to find (you can also set distance_threshold instead).
        :param linkage: str, linkage criterion ('complete', 'average', 'single').
        :param distance_func: a function(interval_a, interval_b) -> distance (scalar).
        :param aggregate: str, how to combine distance across dimensions if needed (e.g., 'mean', 'sum', 'max').
        """
        self.n_clusters = n_clusters
        self.linkage = linkage

        self.model_ = None
        self.labels_ = None
        self.distance_func = distance_func

        if self.distance_func in self.distance_funcs:
            self.isSim = False
        elif self.distance_func in self.similarity_funcs:
            self.isSim = True
        else:
            raise ValueError(f"Unsupported metric: {self.distance_func}")
    
    def compute_distance_matrix(self, intervals):
        if self.isSim:
            dist_matrix = pairwise_similarity(
                intervals, 
                metric=self.distance_func
            )
            dist_matrix = 1 - dist_matrix
        else:
            dist_matrix = pairwise_distance(
                intervals, 
                metric=self.distance_func
            )
        return dist_matrix

    def _compute_centroids(self, intervals, labels, n_clusters):
        """
        Compute the centroids for each cluster
        
        Parameters:
        -----------
        intervals : numpy.ndarray
            Interval data with shape (n_samples, n_dims, 2)
        labels : numpy.ndarray
            Cluster labels with shape (n_samples,)
        n_clusters : int
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
        Fit the hierarchical clustering model to the interval data
        
        Parameters:
        -----------
        intervals: shape (n_samples, n_dims, 2)
            Interval data to cluster
        """
        dist_matrix = self.compute_distance_matrix(intervals)

        self.model_ = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric='precomputed', 
            linkage=self.linkage,
            compute_distances=True
        )
        self.model_.fit(dist_matrix)

        self.labels_ = self.model_.labels_
        self.train_data = intervals
        
        # Calculate and store cluster centroids
        self.centroids_ = self._compute_centroids(intervals, self.labels_, self.n_clusters)

    def get_labels(self):
        if self.labels_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.labels_
    
    def predict(self, intervals_new, method='knn', k=5):
        """
        预测新数据点的聚类标签
        
        :param intervals_new: 形状 (n_samples, n_dims, 2)
        :param method: 预测方法，可选 'knn', 'center', 'nearest'
        :param k: knn方法的邻居数量
        :return: 预测的标签
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
        """使用KNN方法预测"""
        # 计算新数据与训练数据之间的距离
        if self.ifSim:
            cross_dist = cross_similarity(intervals_new, self.train_data, metric = self.distance_func)
            cross_dist = 1 - cross_dist
        else:
            cross_dist = cross_distance(intervals_new, self.train_data, metric = self.distance_func)
        
        # 对每个新样本，找到k个最近的训练样本
        predictions = []
        for i in range(len(intervals_new)):
            # 获取k个最近邻
            nearest_indices = np.argsort(cross_dist[i])[:k]
            nearest_labels = [self.labels_[idx] for idx in nearest_indices]
            
            # 投票选择最常见的标签
            from collections import Counter
            most_common = Counter(nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)
    
    def _predict_centers(self, intervals_new):
        """使用聚类中心预测"""
        # 计算每个聚类的中心
        centers = []
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(self.labels_ == cluster_id)[0]
            cluster_data = self.train_data[cluster_indices]
            
            # 为每个维度分别计算区间的中心
            # 这里我们计算每个维度的下界和上界的平均值，保持区间的形式
            # cluster_data形状: (n_samples, n_dims, 2)
            center = np.mean(cluster_data, axis=0)  # 形状保持为 (n_dims, 2)
            centers.append(center)
        
        centers = np.array(centers)  # 形状: (n_clusters, n_dims, 2)
        
        # 计算到每个中心的距离
        predictions = []
        for interval in intervals_new:
            distances = []
            for center in centers:
                # 使用原始相似度计算方法
                combined = np.stack([interval, center])
                dist_matrix = self.compute_distance_matrix(combined)
                dist = dist_matrix[0, 1]  # 获取交叉距离
                
                distances.append(dist)
            
            # 分配到最近的中心
            closest_center = np.argmin(distances)
            predictions.append(closest_center)
        
        return np.array(predictions)

    def compute_metrics_for_k_range(self, intervals, min_clusters=2, max_clusters=10, 
                       metrics=['distortion'], distance_func=None, 
                       linkage=None):
        """
        Compute evaluation metrics for a range of cluster numbers for hierarchical clustering.
        
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
        distance_func : str, default=None
            Distance function name. If None, uses the current instance's distance function.
        linkage : str, default=None
            Linkage criterion. If None, uses the current instance's linkage.
        random_state : int, default=None
            Random seed (not used in hierarchical clustering but kept for API consistency).
        
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
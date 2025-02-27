import pandas as pd
import numpy as np
from numpy.random import RandomState
from warnings import warn
from sklearn.cluster import AgglomerativeClustering
from interClusLib.similarity_distance import IntervalMetrics
import os

class IntervalAgglomerativeClustering:
    """
    An Agglomerative (Hierarchical) Clustering for interval data (n_dims, 2).
    Uses a precomputed distance matrix from a custom distance function.
    """
    distance_funcs = {"hausdorff", "euclidean", "manhattan"}
    similarity_funcs = {"jaccard", "dice", "bidrectional_min","bidrectional_prod", "marginal"}

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
            dist_matrix = IntervalMetrics.pairwise_similarity(
                intervals, 
                metric=self.distance_func
            )
            dist_matrix = 1 - dist_matrix
        else:
            dist_matrix = IntervalMetrics.pairwise_distance(
                intervals, 
                metric=self.distance_func
            )
        return dist_matrix

    def fit(self, intervals):
        """
        :param intervals: shape (n_samples, n_dims, 2)
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
            cross_dist = IntervalMetrics.cross_similarity(intervals_new, self.train_data, metric = self.distance_func)
            cross_dist = 1 - cross_dist
        else:
            cross_dist = IntervalMetrics.cross_distance(intervals_new, self.train_data, metric = self.distance_func)
        
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
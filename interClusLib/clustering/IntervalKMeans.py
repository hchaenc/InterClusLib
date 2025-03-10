import pandas as pd
import numpy as np
from numpy.random import RandomState
from warnings import warn
from interClusLib.metric import SIMILARITY_FUNCTIONS, DISTANCE_FUNCTIONS
import os

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
        self.random_state = np.random.RandomState(random_state)
        self.train_data = None
        
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

    def predict(self, intervals):
        """
        Assign new data points to the closest cluster.
        """
        predictions = self._assign_clusters(intervals, self.centroids_)
        return predictions

    def get_labels(self):
        if self.labels_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.labels_
        
    def find_optimal_clusters(self, intervals, min_clusters=2, max_clusters=10, 
                              method='l-method', eval_metric='distortion', 
                              visualize=True, **kwargs):
        """
        确定最佳聚类数量
        
        参数:
        intervals: array-like, 形状为(n_samples, n_dims, 2)的区间数据
        min_clusters: int, 最小聚类数
        max_clusters: int, 最大聚类数
        method: str, 确定最佳聚类数的方法，可选值: 'l-method', 'elbow', 'optimize'
        eval_metric: str, 评估指标，可选值取决于已注册的指标
        visualize: bool, 是否可视化结果
        **kwargs: 传递给聚类算法的额外参数
        
        返回:
        int: 最佳聚类数量
        dict: 评估结果
        """
        from interClusLib.cluster_number import metric_registry, l_method, elbow_method, optimize_metric
        import numpy as np
        import matplotlib.pyplot as plt
        
        # 获取评估指标函数
        metric_fn = metric_registry.get_function(eval_metric)
        maximize = metric_registry.should_maximize(eval_metric)
        
        # 保存当前参数
        current_max_iter = self.max_iter
        current_tol = self.tol
        current_distance_func = self.distance_func_name  # 使用保存的距离函数名称
        
        # 计算不同聚类数量下的评估指标
        eval_results = {}
        for k in range(min_clusters, max_clusters + 1):
            # 使用当前类创建模型
            model = self.__class__(
                n_clusters=k,
                max_iter=kwargs.get('max_iter', current_max_iter),
                tol=kwargs.get('tol', current_tol),
                distance_func=kwargs.get('distance_func', current_distance_func),
                random_state=kwargs.get('random_state', 42)
            )
            
            # 训练模型
            model.fit(intervals)
            
            # 根据评估指标函数的需要提供不同的参数
            if eval_metric in ['distortion', 'davies_bouldin', 'calinski_harabasz']:
                # 这些指标需要中心点信息
                score = metric_fn(
                    data=intervals,
                    labels=model.labels_,
                    centers=model.centroids_,
                    metric=current_distance_func,
                )
            else:  # 'silhouette', 'dunn' 或其他指标
                # 这些指标只需要数据和标签
                score = metric_fn(
                    data=intervals,
                    labels=model.labels_,
                    metric=current_distance_func,
                )
            eval_results[k] = score
        
        # 确定最佳聚类数量
        if method.lower() == 'l_method':
            optimal_k = l_method(eval_results, min_clusters, max_clusters)
        elif method.lower() == 'elbow':
            optimal_k = elbow_method(eval_results, min_clusters, max_clusters, convex=maximize)
        elif method.lower() == 'optimize':
            optimal_k = optimize_metric(eval_results, min_clusters, max_clusters, maximize=maximize)
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        # 可视化结果
        if visualize:
            from interClusLib.cluster_number.selector import ClusterNumberSelector
            
            # 创建并配置选择器
            selector = ClusterNumberSelector(min_clusters, max_clusters)
            selector.eval_results = np.array([(k, v) for k, v in sorted(eval_results.items())])
            selector.optimal_k = optimal_k
            
            # 获取指标描述和绘图
            metric_description = metric_registry.get_description(eval_metric)
            plt = selector.plot_evaluation(
                title=f"{method.capitalize()} Method - {metric_description}",
                ylabel=metric_description
            )
            plt.show()
        
        return optimal_k, eval_results
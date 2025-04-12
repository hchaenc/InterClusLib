import unittest
import numpy as np
from interClusLib.metric import SIMILARITY_FUNCTIONS, DISTANCE_FUNCTIONS
from interClusLib.clustering import IntervalKMeans  # 替换为实际路径

class TestIntervalKMeans(unittest.TestCase):
    
    def setUp(self):
        # 创建一些简单的测试数据
        # 数据形状: (n_samples, n_dims, 2) 其中2表示区间的[下界, 上界]
        self.simple_data = np.array([
            # 簇1: 围绕 ([1, 2], [1, 2])
            [[0.8, 1.2], [0.9, 1.1]],  # 样本1：2个维度，每个维度是[下界,上界]
            [[0.9, 1.3], [0.8, 1.2]],  # 样本2
            [[1.1, 1.5], [1.0, 1.4]],  # 样本3
            
            # 簇2: 围绕 ([5, 6], [5, 6])
            [[4.8, 5.2], [4.9, 5.1]],  # 样本4
            [[4.9, 5.3], [4.8, 5.2]],  # 样本5
            [[5.1, 5.5], [5.0, 5.4]],  # 样本6
        ])
        
        # 带有明显簇的随机数据
        rs = np.random.RandomState(42)
        
        # 创建第一个簇 - 20个样本，每个样本2个维度
        cluster1 = np.zeros((20, 2, 2))  # (样本数, 维度数, 区间边界数)
        for i in range(20):
            for j in range(2):  # 2个维度
                # 生成0-2范围内的随机区间
                lower = rs.uniform(0, 1.5)
                upper = rs.uniform(lower, 2.0)  # 确保上界大于下界
                cluster1[i, j] = [lower, upper]
        
        # 创建第二个簇 - 20个样本，每个样本2个维度
        cluster2 = np.zeros((20, 2, 2))
        for i in range(20):
            for j in range(2):  # 2个维度
                # 生成8-10范围内的随机区间
                lower = rs.uniform(8, 9.5)
                upper = rs.uniform(lower, 10.0)  # 确保上界大于下界
                cluster2[i, j] = [lower, upper]
        
        # 合并两个簇
        self.random_data = np.vstack([cluster1, cluster2])
    
    def test_initialization(self):
        """测试初始化参数是否正确设置"""
        kmeans = IntervalKMeans(n_clusters=3, max_iter=200, tol=1e-5, distance_func='euclidean')
        self.assertEqual(kmeans.n_clusters, 3)
        self.assertEqual(kmeans.max_iter, 200)
        self.assertEqual(kmeans.tol, 1e-5)
        self.assertEqual(kmeans.distance_func, 'euclidean')
        self.assertFalse(kmeans.isSim)  # euclidean是距离函数，不是相似度函数
    
    def test_invalid_distance_function(self):
        """测试无效的距离函数是否抛出预期的异常"""
        with self.assertRaises(ValueError):
            IntervalKMeans(distance_func='invalid_function')
    
    def test_init_centroids(self):
        """测试中心点初始化"""
        kmeans = IntervalKMeans(n_clusters=2)
        centroids = kmeans._init_centroids(self.simple_data)
        self.assertEqual(centroids.shape, (2, 2, 2))  # 2个聚类，1个维度，每个区间2个值
        # 确认中心点是从原始数据中选择的
        for centroid in centroids:
            self.assertTrue(any(np.array_equal(centroid, sample) for sample in self.simple_data))
    
    def test_compute_centroid(self):
        """测试中心点计算"""
        kmeans = IntervalKMeans()
        cluster_data = np.array([
            [[[1, 3], [2, 4]]],
            [[[2, 4], [3, 5]]]
        ])
        centroid = kmeans._compute_centroid(cluster_data)
        expected = np.array([[[1.5, 3.5], [2.5, 4.5]]])
        np.testing.assert_array_almost_equal(centroid, expected)
    
    def test_assign_clusters(self):
        """测试样本到聚类的分配"""
        kmeans = IntervalKMeans(n_clusters=2, distance_func='euclidean')
        
        centroids = np.array([
            [[1, 2], [1, 2]], 
            [[5, 6], [5, 6]]
        ])
        
        labels = kmeans._assign_clusters(self.simple_data, centroids)
        # 前3个样本应该分配给第一个聚类，后3个样本应该分配给第二个聚类
        expected_labels = np.array([0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(labels, expected_labels)
    
    def test_fit_simple_data(self):
        """测试聚类算法在简单数据上的拟合"""
        kmeans = IntervalKMeans(n_clusters=2, distance_func='euclidean')
        kmeans.fit(self.simple_data)
        
        # 检查聚类标签是否合理划分
        # 我们期望前三个点在一个聚类，后三个点在另一个聚类
        labels = kmeans.get_labels()
        # 因为初始化是随机的，我们不能直接比较标签值，而是检查相似样本的标签是否相同
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertEqual(labels[3], labels[5])
        self.assertNotEqual(labels[0], labels[3])
    
    def test_fit_random_data(self):
        """测试聚类算法在随机数据上的拟合"""
        kmeans = IntervalKMeans(n_clusters=2, distance_func='euclidean')
        kmeans.fit(self.random_data)
        
        # 验证聚类结果
        labels = kmeans.get_labels()
        # 检查簇大小是否合理 (接近各20个样本)
        unique, counts = np.unique(labels, return_counts=True)
        self.assertEqual(len(unique), 2)  # 应该有两个唯一的标签
        
        # 检查每个簇内的样本是否相似
        centroids = kmeans.centroids_
        
        # 计算每个样本到其分配的中心点的距离
        intra_cluster_distances = []
        for i, sample in enumerate(self.random_data):
            cluster_idx = labels[i]
            centroid = centroids[cluster_idx]
            dist = np.sum((sample - centroid)**2)  # 简单欧氏距离平方
            intra_cluster_distances.append(dist)
        
        # 检查簇内距离是否小于簇间距离
        avg_intra_cluster_dist = np.mean(intra_cluster_distances)
        inter_cluster_dist = np.sum((centroids[0] - centroids[1])**2)
        self.assertLess(avg_intra_cluster_dist, inter_cluster_dist)
    
    def test_compute_metrics(self):
        """测试度量计算功能"""
        kmeans = IntervalKMeans(n_clusters=2)
        
        # 简单数据上测试
        metrics = kmeans.compute_metrics_for_k_range(
            self.simple_data, 
            min_clusters=2, 
            max_clusters=3, 
            metrics=['distortion']
        )
        
        # 验证结果格式
        self.assertIn('distortion', metrics)
        self.assertIn(2, metrics['distortion'])
        self.assertIn(3, metrics['distortion'])
        
        # 测试错误的度量名称
        with self.assertRaises(ValueError):
            kmeans.compute_metrics_for_k_range(
                self.simple_data, 
                metrics=['invalid_metric']
            )
    
    def test_random_state_reproducibility(self):
        """测试随机种子是否能产生可复现的结果"""
        # 使用相同的随机种子
        kmeans1 = IntervalKMeans(n_clusters=2, random_state=42)
        kmeans1.fit(self.random_data)
        
        kmeans2 = IntervalKMeans(n_clusters=2, random_state=42)
        kmeans2.fit(self.random_data)
        
        # 结果应该相同
        np.testing.assert_array_equal(kmeans1.labels_, kmeans2.labels_)
        np.testing.assert_array_almost_equal(kmeans1.centroids_, kmeans2.centroids_)
        
        # 使用不同的随机种子
        kmeans3 = IntervalKMeans(n_clusters=2, random_state=0)
        kmeans3.fit(self.random_data)
        
        # 结果可能不同（不一定，但很可能）
        # 我们不直接断言它们不同，因为随机性可能导致相同的结果
    
    def test_cluster_and_return(self):
        """测试cluster_and_return方法"""
        kmeans = IntervalKMeans(n_clusters=3)  # 默认k=3
        labels, centroids = kmeans.cluster_and_return(self.simple_data, k=2)  # 覆盖为k=2
        
        # 检查返回值的形状
        self.assertEqual(len(labels), len(self.simple_data))
        self.assertEqual(centroids.shape[0], 2)  # 2个聚类
        
        # 检查标签是否合理
        unique_labels = np.unique(labels)
        self.assertTrue(all(label in [0, 1] for label in unique_labels))

if __name__ == '__main__':
    unittest.main()
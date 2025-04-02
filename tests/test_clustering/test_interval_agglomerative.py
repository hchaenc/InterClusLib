import unittest
import numpy as np
from interClusLib.metric import pairwise_distance, pairwise_similarity
from interClusLib.clustering import IntervalAgglomerativeClustering

class TestIntervalAgglomerativeClustering(unittest.TestCase):
    
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
        hclust = IntervalAgglomerativeClustering(n_clusters=3, linkage='complete', distance_func='euclidean')
        self.assertEqual(hclust.n_clusters, 3)
        self.assertEqual(hclust.linkage, 'complete')
        self.assertEqual(hclust.distance_func, 'euclidean')
        self.assertFalse(hclust.isSim)  # euclidean是距离函数，不是相似度函数
        
        # 测试相似度度量
        hclust2 = IntervalAgglomerativeClustering(n_clusters=2, distance_func='jaccard')
        self.assertTrue(hclust2.isSim)  # jaccard是相似度函数
    
    def test_invalid_parameters(self):
        """测试无效的参数是否抛出预期的异常"""
        # 测试无效的连接方法
        with self.assertRaises(ValueError):
            IntervalAgglomerativeClustering(linkage='invalid_linkage')
        
        # 测试无效的距离函数
        with self.assertRaises(ValueError):
            IntervalAgglomerativeClustering(distance_func='invalid_function')
    
    def test_compute_distance_matrix(self):
        """测试距离矩阵计算"""
        # 使用欧氏距离
        hclust = IntervalAgglomerativeClustering(distance_func='euclidean')
        dist_matrix = hclust.compute_distance_matrix(self.simple_data)
        
        # 验证距离矩阵的形状和属性
        self.assertEqual(dist_matrix.shape, (6, 6))  # 6个样本
        self.assertTrue(np.allclose(dist_matrix, dist_matrix.T))  # 对称矩阵
        self.assertTrue(np.all(np.diag(dist_matrix) == 0))  # 对角线为0
        
        # 检查不同簇的样本之间的距离是否大于同一簇内的距离
        # 簇1: 样本0,1,2  簇2: 样本3,4,5
        within_cluster1_avg = (dist_matrix[0, 1] + dist_matrix[0, 2] + dist_matrix[1, 2]) / 3
        within_cluster2_avg = (dist_matrix[3, 4] + dist_matrix[3, 5] + dist_matrix[4, 5]) / 3
        between_clusters_avg = (dist_matrix[0, 3] + dist_matrix[0, 4] + dist_matrix[0, 5] + 
                              dist_matrix[1, 3] + dist_matrix[1, 4] + dist_matrix[1, 5] + 
                              dist_matrix[2, 3] + dist_matrix[2, 4] + dist_matrix[2, 5]) / 9
        
        self.assertLess(within_cluster1_avg, between_clusters_avg)
        self.assertLess(within_cluster2_avg, between_clusters_avg)
        
        # 使用相似度度量
        hclust_sim = IntervalAgglomerativeClustering(distance_func='jaccard')
        dist_matrix_sim = hclust_sim.compute_distance_matrix(self.simple_data)
        
        # 验证转换为距离后的矩阵属性（相似度转距离: 1-相似度）
        self.assertEqual(dist_matrix_sim.shape, (6, 6))
        self.assertTrue(np.allclose(dist_matrix_sim, dist_matrix_sim.T))
        self.assertTrue(np.all(np.diag(dist_matrix_sim) == 0))
    
    def test_compute_centroids(self):
        """测试中心点计算"""
        hclust = IntervalAgglomerativeClustering(n_clusters=2)
        
        # 创建人工标签：前3个样本是簇0，后3个样本是簇1
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        centroids = hclust._compute_centroids(self.simple_data, labels, 2)
        
        # 验证中心点的形状和内容
        self.assertEqual(centroids.shape, (2, 2, 2))  # 2个聚类，2个维度，每个区间2个值
        
        # 手动计算预期的中心点
        expected_centroid0 = np.mean(self.simple_data[:3], axis=0)
        expected_centroid1 = np.mean(self.simple_data[3:], axis=0)
        
        np.testing.assert_array_almost_equal(centroids[0], expected_centroid0)
        np.testing.assert_array_almost_equal(centroids[1], expected_centroid1)
        
        # 测试空簇的处理（应该返回零数组）
        # 创建一个标签数组，其中没有分配给簇1的样本
        empty_cluster_labels = np.zeros(len(self.simple_data))
        
        # 使用with语句捕获打印的警告消息
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            empty_centroids = hclust._compute_centroids(self.simple_data, empty_cluster_labels, 2)
        
        # 验证警告消息和结果
        self.assertIn("Warning: Cluster 1 is empty", f.getvalue())
        self.assertEqual(empty_centroids.shape, (2, 2, 2))
        np.testing.assert_array_almost_equal(empty_centroids[0], np.mean(self.simple_data, axis=0))
        np.testing.assert_array_equal(empty_centroids[1], np.zeros((2, 2)))
    
    def test_fit_simple_data(self):
        """测试聚类算法在简单数据上的拟合"""
        hclust = IntervalAgglomerativeClustering(n_clusters=2, distance_func='euclidean')
        hclust.fit(self.simple_data)
        
        # 检查属性是否已正确设置
        self.assertEqual(hclust.n_samples_, 6)
        self.assertIsNotNone(hclust.linkage_matrix_)
        self.assertIsNotNone(hclust.labels_)
        self.assertIsNotNone(hclust.centroids_)
        
        # 检查聚类标签
        labels = hclust.get_labels()
        self.assertEqual(len(labels), 6)
        
        # 对于层次聚类，我们期望相似的样本有相同的标签
        # 由于标签的值可能会变，我们检查结构而不是具体的值
        # 我们期望前三个点在同一个簇，后三个点在另一个簇
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertEqual(labels[3], labels[5])
        self.assertNotEqual(labels[0], labels[3])
    
    def test_get_labels(self):
        """测试获取标签"""
        hclust = IntervalAgglomerativeClustering(n_clusters=2)
        
        # 未拟合时应该抛出异常
        with self.assertRaises(RuntimeError):
            hclust.get_labels()
        
        # 拟合后应该返回标签
        hclust.fit(self.simple_data)
        labels = hclust.get_labels()
        self.assertEqual(len(labels), 6)
        self.assertTrue(np.all(np.isin(labels, [0, 1])))  # 所有标签都是0或1
    
    def test_get_dendrogram_data(self):
        """测试获取树状图数据"""
        hclust = IntervalAgglomerativeClustering(n_clusters=2)
        
        # 未拟合时应该抛出异常
        with self.assertRaises(RuntimeError):
            hclust.get_dendrogram_data()
        
        # 拟合后应该返回树状图数据
        hclust.fit(self.simple_data)
        dendrogram_data = hclust.get_dendrogram_data()
        
        self.assertIn('linkage_matrix', dendrogram_data)
        self.assertIn('labels', dendrogram_data)
        self.assertIn('n_leaves', dendrogram_data)
        
        self.assertEqual(dendrogram_data['n_leaves'], 6)  # 6个叶子节点（样本）
        self.assertEqual(len(dendrogram_data['labels']), 6)  # 6个标签
        self.assertEqual(dendrogram_data['linkage_matrix'].shape, (5, 4))  # n-1行的连接矩阵
    
    def test_compute_metrics_for_k_range(self):
        """测试度量计算功能"""
        hclust = IntervalAgglomerativeClustering(n_clusters=2)
        
        # 简单数据上测试
        metrics = hclust.compute_metrics_for_k_range(
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
            hclust.compute_metrics_for_k_range(
                self.simple_data, 
                metrics=['invalid_metric']
            )
    
    def test_fit_random_data(self):
        """测试聚类算法在随机数据上的拟合"""
        hclust = IntervalAgglomerativeClustering(n_clusters=2, distance_func='euclidean')
        hclust.fit(self.random_data)
        
        # 验证聚类结果
        labels = hclust.get_labels()
        
        # 检查簇大小是否合理（接近各20个样本）
        unique, counts = np.unique(labels, return_counts=True)
        self.assertEqual(len(unique), 2)  # 应该有两个唯一的标签
        
        # 检查每个簇内的样本是否相似
        centroids = hclust.centroids_
        
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
    
    def test_different_linkage_methods(self):
        """测试不同的连接方法"""
        linkage_methods = ['single', 'complete', 'average', 'ward']
        
        for method in linkage_methods:
            hclust = IntervalAgglomerativeClustering(n_clusters=2, linkage=method)
            hclust.fit(self.simple_data)
            
            # 检查是否正确拟合
            self.assertIsNotNone(hclust.labels_)
            self.assertIsNotNone(hclust.centroids_)
            
            # 不同的连接方法应该都能识别出两个明显的簇
            labels = hclust.get_labels()
            # 我们期望前三个点在同一个簇，后三个点在另一个簇
            self.assertEqual(labels[0], labels[1])
            self.assertEqual(labels[0], labels[2])
            self.assertEqual(labels[3], labels[4])
            self.assertEqual(labels[3], labels[5])
            self.assertNotEqual(labels[0], labels[3])
    
    def test_cluster_and_return(self):
        """测试cluster_and_return方法"""
        hclust = IntervalAgglomerativeClustering(n_clusters=3)  # 默认k=3
        labels, centroids = hclust.cluster_and_return(self.simple_data, k=2)  # 覆盖为k=2
        
        # 检查返回值的形状
        self.assertEqual(len(labels), len(self.simple_data))
        self.assertEqual(centroids.shape[0], 2)  # 2个聚类
        
        # 检查标签是否合理
        unique_labels = np.unique(labels)
        self.assertTrue(all(label in [0, 1] for label in unique_labels))

if __name__ == '__main__':
    unittest.main()
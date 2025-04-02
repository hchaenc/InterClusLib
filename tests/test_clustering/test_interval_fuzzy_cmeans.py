import unittest
import numpy as np
from interClusLib.metric import SIMILARITY_FUNCTIONS, DISTANCE_FUNCTIONS
from interClusLib.clustering import IntervalFuzzyCMeans

class TestIntervalFuzzyCMeans(unittest.TestCase):
    
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
        fcm = IntervalFuzzyCMeans(n_clusters=3, m=2.5, max_iter=200, tol=1e-5, 
                                  adaptive_weights=True, distance_func='euclidean')
        self.assertEqual(fcm.n_clusters, 3)
        self.assertEqual(fcm.m, 2.5)
        self.assertEqual(fcm.max_iter, 200)
        self.assertEqual(fcm.tol, 1e-5)
        self.assertTrue(fcm.adaptive_weights)
        self.assertEqual(fcm.distance_func, 'euclidean')
        self.assertFalse(fcm.isSim)  # euclidean是距离函数，不是相似度函数
        
        # 测试相似度度量
        fcm2 = IntervalFuzzyCMeans(n_clusters=2, distance_func='jaccard')
        self.assertTrue(fcm2.isSim)  # jaccard是相似度函数
    
    def test_invalid_distance_function(self):
        """测试无效的距离函数是否抛出预期的异常"""
        with self.assertRaises(ValueError):
            IntervalFuzzyCMeans(distance_func='invalid_function')
    
    def test_init_membership(self):
        """测试隶属度矩阵初始化"""
        fcm = IntervalFuzzyCMeans(n_clusters=3)
        n_samples = 10
        U = fcm._init_membership(n_samples)
        
        # 检查形状
        self.assertEqual(U.shape, (n_samples, 3))
        
        # 检查每行和是否为1
        row_sums = U.sum(axis=1)
        np.testing.assert_almost_equal(row_sums, np.ones(n_samples))
        
        # 检查值是否在[0,1]范围内
        self.assertTrue(np.all(U >= 0))
        self.assertTrue(np.all(U <= 1))
    
    def test_update_centers(self):
        """测试中心点更新"""
        fcm = IntervalFuzzyCMeans(n_clusters=2, m=2.0)
        # 创建一个简单的U
        fcm.U = np.array([
            [1.0, 0.0],  # 样本0完全属于簇0
            [1.0, 0.0],  # 样本1完全属于簇0
            [1.0, 0.0],  # 样本2完全属于簇0
            [0.0, 1.0],  # 样本3完全属于簇1
            [0.0, 1.0],  # 样本4完全属于簇1
            [0.0, 1.0]   # 样本5完全属于簇1
        ])
        
        fcm._update_centers(self.simple_data)
        
        # 检查形状
        self.assertEqual(fcm.centers_a.shape, (2, 2))  # 2个簇，2个维度
        self.assertEqual(fcm.centers_b.shape, (2, 2))  # 2个簇，2个维度
        
        # 验证中心点计算
        # 簇0应该是前3个样本的平均值
        expected_center0_a = np.mean(self.simple_data[:3, :, 0], axis=0)
        expected_center0_b = np.mean(self.simple_data[:3, :, 1], axis=0)
        # 簇1应该是后3个样本的平均值
        expected_center1_a = np.mean(self.simple_data[3:, :, 0], axis=0)
        expected_center1_b = np.mean(self.simple_data[3:, :, 1], axis=0)
        
        np.testing.assert_array_almost_equal(fcm.centers_a[0], expected_center0_a)
        np.testing.assert_array_almost_equal(fcm.centers_b[0], expected_center0_b)
        np.testing.assert_array_almost_equal(fcm.centers_a[1], expected_center1_a)
        np.testing.assert_array_almost_equal(fcm.centers_b[1], expected_center1_b)
    
    def test_convert_centers_to_intervals(self):
        """测试中心点转换为区间格式"""
        fcm = IntervalFuzzyCMeans(n_clusters=2)
        fcm.centers_a = np.array([[1, 2], [3, 4]])
        fcm.centers_b = np.array([[5, 6], [7, 8]])
        
        intervals = fcm._convert_centers_to_intervals()
        
        # 检查形状
        self.assertEqual(intervals.shape, (2, 2, 2))  # 2个簇，2个维度，每个区间2个值
        
        # 验证结果
        expected = np.array([
            [[1, 5], [2, 6]],  # 簇0: [1,5], [2,6]
            [[3, 7], [4, 8]]   # 簇1: [3,7], [4,8]
        ])
        np.testing.assert_array_equal(intervals, expected)
    
    def test_compute_distance(self):
        """测试距离计算"""
        # 使用欧氏距离
        fcm = IntervalFuzzyCMeans(distance_func='euclidean')
        
        # 使用简单的区间
        x_k = np.array([[1, 2], [3, 4]])  # 样本，2个维度
        c_i = np.array([[5, 6], [7, 8]])  # 中心点，2个维度
        
        distances = fcm._compute_distance(x_k, c_i)
        
        # 检查形状
        self.assertEqual(distances.shape, (2,))  # 2个维度
        
        # 验证计算 (假设euclidean计算正确)
        # 我们只需要检查调用逻辑，而不是具体的数值
        
        # 使用相似度度量
        fcm_sim = IntervalFuzzyCMeans(distance_func='jaccard')
        distances_sim = fcm_sim._compute_distance(x_k, c_i)
        
        # 检查形状
        self.assertEqual(distances_sim.shape, (2,))  # 2个维度
    
    def test_update_membership(self):
        """测试隶属度矩阵更新"""
        fcm = IntervalFuzzyCMeans(n_clusters=2, m=2.0)
        
        # 创建一个简单的距离矩阵
        # 假设样本0,1,2靠近簇0，样本3,4,5靠近簇1
        distances = np.zeros((6, 2, 2))  # (n_samples, n_clusters, n_dims)
        # 簇0
        distances[0, 0, :] = [0.1, 0.2]  # 样本0到簇0的距离 (每个维度)
        distances[0, 1, :] = [5.0, 5.1]  # 样本0到簇1的距离
        distances[1, 0, :] = [0.2, 0.3]
        distances[1, 1, :] = [4.9, 5.0]
        distances[2, 0, :] = [0.3, 0.4]
        distances[2, 1, :] = [4.8, 4.9]
        # 簇1
        distances[3, 0, :] = [4.7, 4.8]
        distances[3, 1, :] = [0.4, 0.5]
        distances[4, 0, :] = [4.6, 4.7]
        distances[4, 1, :] = [0.5, 0.6]
        distances[5, 0, :] = [4.5, 4.6]
        distances[5, 1, :] = [0.6, 0.7]
        
        # 测试非自适应权重
        fcm.adaptive_weights = False
        fcm.U = np.ones((6, 2)) / 2  # 初始均匀分布
        U_new = fcm._update_membership(distances)
        
        # 检查形状
        self.assertEqual(U_new.shape, (6, 2))
        
        # 检查每行和是否为1
        row_sums = U_new.sum(axis=1)
        np.testing.assert_almost_equal(row_sums, np.ones(6))
        
        # 样本0,1,2应该主要属于簇0，样本3,4,5应该主要属于簇1
        self.assertTrue(U_new[0, 0] > 0.8)  # 样本0主要属于簇0
        self.assertTrue(U_new[1, 0] > 0.8)
        self.assertTrue(U_new[2, 0] > 0.8)
        self.assertTrue(U_new[3, 1] > 0.8)  # 样本3主要属于簇1
        self.assertTrue(U_new[4, 1] > 0.8)
        self.assertTrue(U_new[5, 1] > 0.8)
        
        # 测试自适应权重
        fcm.adaptive_weights = True
        fcm.k = np.ones((2, 2))  # 均匀权重
        fcm.U = np.ones((6, 2)) / 2  # 重置隶属度
        U_new_adaptive = fcm._update_membership(distances)
        
        # 验证自适应权重产生了相似的效果
        self.assertEqual(U_new_adaptive.shape, (6, 2))
        np.testing.assert_almost_equal(U_new_adaptive.sum(axis=1), np.ones(6))
        self.assertTrue(U_new_adaptive[0, 0] > 0.8)
        self.assertTrue(U_new_adaptive[3, 1] > 0.8)
    
    def test_compute_adaptive_weights(self):
        """测试自适应权重计算（仅当adaptive_weights=True）"""
        fcm = IntervalFuzzyCMeans(n_clusters=2, m=2.0, adaptive_weights=True)
        
        # 设置隶属度矩阵
        fcm.U = np.array([
            [0.9, 0.1],  # 样本0主要属于簇0
            [0.9, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],  # 样本3主要属于簇1
            [0.1, 0.9],
            [0.1, 0.9]
        ])
        
        # 创建距离矩阵
        distances = np.ones((6, 2, 2))  # 初始化为1
        # 给簇0的样本在维度0中较小的距离
        distances[:3, 0, 0] = 0.2
        # 给簇1的样本在维度1中较小的距离
        distances[3:, 1, 1] = 0.2
        
        fcm._compute_adaptive_weights(distances)
        
        # 检查形状
        self.assertEqual(fcm.k.shape, (2, 2))
        
        # 簇0应该在维度0中有较大的权重，簇1应该在维度1中有较大的权重
        self.assertTrue(fcm.k[0, 0] > fcm.k[0, 1])
        self.assertTrue(fcm.k[1, 1] > fcm.k[1, 0])
    
    def test_compute_objective(self):
        """测试目标函数计算"""
        fcm = IntervalFuzzyCMeans(n_clusters=2, m=2.0)
        
        # 设置隶属度矩阵
        fcm.U = np.array([
            [1.0, 0.0],  # 样本0完全属于簇0
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],  # 样本3完全属于簇1
            [0.0, 1.0],
            [0.0, 1.0]
        ])
        
        # 创建距离矩阵，所有距离都是1
        distances = np.ones((6, 2, 2))
        
        # 非自适应权重
        fcm.adaptive_weights = False
        obj1 = fcm._compute_objective(distances)
        
        # 验证结果: J = 3*1*2 + 3*1*2 = 12 (3个样本每个簇，每个样本有2个维度，距离都是1)
        self.assertAlmostEqual(obj1, 12.0)
        
        # 自适应权重
        fcm.adaptive_weights = True
        fcm.k = np.array([[0.5, 0.5], [0.5, 0.5]])  # 均匀权重
        obj2 = fcm._compute_objective(distances)
        
        # 验证结果应该是之前的一半: J = 3*1*2*0.5 + 3*1*2*0.5 = 6
        self.assertAlmostEqual(obj2, 6.0)
    
    def test_fit_simple_data(self):
        """测试聚类算法在简单数据上的拟合"""
        fcm = IntervalFuzzyCMeans(n_clusters=2, distance_func='euclidean', max_iter=100)
        fcm.fit(self.simple_data)
        
        # 检查属性是否已正确设置
        self.assertIsNotNone(fcm.U)
        self.assertIsNotNone(fcm.centers_a)
        self.assertIsNotNone(fcm.centers_b)
        self.assertIsNotNone(fcm.objective_)
        self.assertIsNotNone(fcm.centroids_)
        
        # 检查隶属度矩阵
        U = fcm.get_membership()
        self.assertEqual(U.shape, (6, 2))  # 6个样本，2个簇
        self.assertTrue(np.all(U >= 0) and np.all(U <= 1))  # 值在[0,1]范围内
        np.testing.assert_almost_equal(U.sum(axis=1), np.ones(6))  # 每行和为1
        
        # 获取硬聚类结果
        labels = fcm.get_crisp_assignments()
        self.assertEqual(len(labels), 6)
        
        # 验证聚类结果：预期前3个样本为一组，后3个样本为另一组
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertEqual(labels[3], labels[5])
        self.assertNotEqual(labels[0], labels[3])
    
    def test_fit_with_adaptive_weights(self):
        """测试带自适应权重的拟合"""
        fcm = IntervalFuzzyCMeans(n_clusters=2, distance_func='euclidean', 
                                  adaptive_weights=True, max_iter=100)
        fcm.fit(self.simple_data)
        
        # 检查额外的自适应权重属性
        self.assertIsNotNone(fcm.k)
        self.assertEqual(fcm.k.shape, (2, 2))  # 2个簇，2个维度的权重
        
        # 获取硬聚类结果
        labels = fcm.get_crisp_assignments()
        
        # 验证聚类结果：预期前3个样本为一组，后3个样本为另一组
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertEqual(labels[3], labels[5])
        self.assertNotEqual(labels[0], labels[3])
    
    def test_get_membership(self):
        """测试获取隶属度矩阵"""
        fcm = IntervalFuzzyCMeans(n_clusters=2)
        
        # 未拟合时应该抛出异常
        with self.assertRaises(RuntimeError):
            fcm.get_membership()
        
        # 拟合后应该返回隶属度矩阵
        fcm.fit(self.simple_data)
        U = fcm.get_membership()
        self.assertEqual(U.shape, (6, 2))
        self.assertTrue(np.all(U >= 0) and np.all(U <= 1))  # 值在[0,1]范围内
        np.testing.assert_almost_equal(U.sum(axis=1), np.ones(6))  # 每行和为1
    
    def test_get_centers(self):
        """测试获取中心点"""
        fcm = IntervalFuzzyCMeans(n_clusters=2)
        
        # 未拟合时应该抛出异常
        with self.assertRaises(RuntimeError):
            fcm.get_centers()
        
        # 拟合后应该返回中心点
        fcm.fit(self.simple_data)
        centers_a, centers_b = fcm.get_centers()
        self.assertEqual(centers_a.shape, (2, 2))  # 2个簇，2个维度
        self.assertEqual(centers_b.shape, (2, 2))  # 2个簇，2个维度
    
    def test_get_objective(self):
        """测试获取目标函数值"""
        fcm = IntervalFuzzyCMeans(n_clusters=2)
        
        # 拟合前调用不应该有效值
        self.assertIsNone(fcm.get_objective())
        
        # 拟合后应该返回目标函数值
        fcm.fit(self.simple_data)
        obj = fcm.get_objective()
        self.assertIsNotNone(obj)
        self.assertGreater(obj, 0)  # 目标函数值应该是正数
    
    def test_get_crisp_assignments(self):
        """测试获取硬聚类结果"""
        fcm = IntervalFuzzyCMeans(n_clusters=2)
        
        # 未拟合时应该抛出异常
        with self.assertRaises(RuntimeError):
            fcm.get_crisp_assignments()
        
        # 拟合后应该返回硬聚类结果
        fcm.fit(self.simple_data)
        labels = fcm.get_crisp_assignments()
        self.assertEqual(len(labels), 6)
        self.assertTrue(np.all(np.isin(labels, [0, 1])))  # 所有标签都是0或1
    
    def test_compute_metrics_for_k_range(self):
        """测试度量计算功能"""
        fcm = IntervalFuzzyCMeans(n_clusters=2)

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        
        # 简单数据上测试
        with redirect_stdout(f):
            metrics = fcm.compute_metrics_for_k_range(
                self.simple_data, 
                min_clusters=2, 
                max_clusters=3, 
                metrics=['distortion']
            )

        output = f.getvalue()
        print(f"捕获的输出: {output}")
        print(f"metrics结果: {metrics}")
        
        # 验证结果格式
        self.assertIn('distortion', metrics)
        self.assertIn(2, metrics['distortion'])
        self.assertIn(3, metrics['distortion'])
        
        # 测试错误的度量名称
        with self.assertRaises(ValueError):
            fcm.compute_metrics_for_k_range(
                self.simple_data, 
                metrics=['invalid_metric']
            )
    
    def test_fit_random_data(self):
        """测试聚类算法在随机数据上的拟合"""
        fcm = IntervalFuzzyCMeans(n_clusters=2, distance_func='euclidean')
        fcm.fit(self.random_data)
        
        # 验证聚类结果
        labels = fcm.get_crisp_assignments()
        
        # 检查簇大小是否合理（接近各20个样本）
        unique, counts = np.unique(labels, return_counts=True)
        self.assertEqual(len(unique), 2)  # 应该有两个唯一的标签
        
        # 检查每个簇内的样本是否相似
        centroids = fcm.centroids_
        
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
    
    def test_cluster_and_return(self):
        """测试cluster_and_return方法"""
        fcm = IntervalFuzzyCMeans(n_clusters=3)  # 默认k=3
        labels, centroids = fcm.cluster_and_return(self.simple_data, k=2)  # 覆盖为k=2
        
        # 检查返回值的形状
        self.assertEqual(len(labels), len(self.simple_data))
        self.assertEqual(centroids.shape[0], 2)  # 2个聚类
        
        # 检查标签是否合理
        unique_labels = np.unique(labels)
        self.assertTrue(all(label in [0, 1] for label in unique_labels))
    
    def test_different_fuzzifier(self):
        """测试不同的模糊因子m"""
        # 较小的m (更接近硬聚类)
        fcm_low = IntervalFuzzyCMeans(n_clusters=2, m=1.1)
        fcm_low.fit(self.simple_data)
        U_low = fcm_low.get_membership()
        
        # 较大的m (更模糊的聚类)
        fcm_high = IntervalFuzzyCMeans(n_clusters=2, m=3.0)
        fcm_high.fit(self.simple_data)
        U_high = fcm_high.get_membership()
        
        # 较小的m应该产生更极端的隶属度（更接近0或1）
        max_membership_low = np.max(U_low, axis=1)
        max_membership_high = np.max(U_high, axis=1)
        
        # 验证m较小时隶属度更极端
        self.assertTrue(np.mean(max_membership_low) > np.mean(max_membership_high))

if __name__ == '__main__':
    unittest.main()
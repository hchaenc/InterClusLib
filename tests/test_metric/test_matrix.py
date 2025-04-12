import unittest
import numpy as np
from interClusLib.metric.matrix import (
    pairwise_similarity,
    pairwise_distance,
    cross_similarity,
    cross_distance
)

class TestPairwiseMetrics(unittest.TestCase):
    """Test suite for pairwise and cross metric calculations"""
    
    def setUp(self):
        """Set up test data"""
        # Create a small set of 1D intervals
        self.intervals_1d = np.array([
            [[1.0, 3.0]],  # First interval
            [[2.0, 4.0]],  # Second interval
            [[5.0, 7.0]]   # Third interval
        ])
        
        # Create a small set of 2D intervals
        self.intervals_2d = np.array([
            [[1.0, 3.0], [1.0, 3.0]],  # First interval
            [[2.0, 4.0], [2.0, 4.0]],  # Second interval
            [[5.0, 7.0], [5.0, 7.0]]   # Third interval
        ])
        
        # Create a second set of intervals for cross-calculations
        self.intervals_b_2d = np.array([
            [[0.0, 2.0], [0.0, 2.0]],  # First interval
            [[3.0, 5.0], [3.0, 5.0]]   # Second interval
        ])

    def test_pairwise_similarity_shape(self):
        """Test that pairwise_similarity returns correct shape"""
        # For 1D intervals
        sim_matrix = pairwise_similarity(self.intervals_1d)
        self.assertEqual(sim_matrix.shape, (3, 3))
        
        # For 2D intervals
        sim_matrix = pairwise_similarity(self.intervals_2d)
        self.assertEqual(sim_matrix.shape, (3, 3))

    def test_pairwise_similarity_values(self):
        """Test that pairwise_similarity computes correct values"""
        # Using Jaccard similarity
        sim_matrix = pairwise_similarity(self.intervals_1d, metric="jaccard")
        
        # Diagonal should be 1.0 (self-similarity)
        self.assertEqual(sim_matrix[0, 0], 1.0)
        self.assertEqual(sim_matrix[1, 1], 1.0)
        self.assertEqual(sim_matrix[2, 2], 1.0)
        
        # Check specific pairs
        # Intervals [1,3] and [2,4] have intersection [2,3] = 1, union [1,4] = 3
        # Jaccard = 1/3 ≈ 0.333
        self.assertAlmostEqual(sim_matrix[0, 1], 1/3, places=5)
        self.assertAlmostEqual(sim_matrix[1, 0], 1/3, places=5)
        
        # Intervals [1,3] and [5,7] don't overlap, so Jaccard = 0
        self.assertEqual(sim_matrix[0, 2], 0.0)
        self.assertEqual(sim_matrix[2, 0], 0.0)

    def test_pairwise_similarity_metrics(self):
        """Test pairwise_similarity with different metrics"""
        # Test with Dice similarity
        sim_matrix_dice = pairwise_similarity(self.intervals_1d, metric="dice")
        
        # Intervals [1,3] and [2,4] have intersection [2,3] = 1, sum = 2+2 = 4
        # Dice = 2*1/4 = 0.5
        self.assertAlmostEqual(sim_matrix_dice[0, 1], 0.5, places=5)
        
        # Test with bidirectional_min
        sim_matrix_bidir = pairwise_similarity(self.intervals_1d, metric="bidirectional_min")
        
        # For intervals [1,3] and [2,4]:
        # Overlap of [1,3] is 1/2 = 0.5
        # Overlap of [2,4] is 1/2 = 0.5
        # Min is 0.5
        self.assertAlmostEqual(sim_matrix_bidir[0, 1], 0.5, places=5)

    def test_pairwise_similarity_invalid_metric(self):
        """Test pairwise_similarity with invalid metric"""
        with self.assertRaises(ValueError):
            pairwise_similarity(self.intervals_1d, metric="invalid_metric")

    def test_pairwise_distance_shape(self):
        """Test that pairwise_distance returns correct shape"""
        # For 1D intervals
        dist_matrix = pairwise_distance(self.intervals_1d)
        self.assertEqual(dist_matrix.shape, (3, 3))
        
        # For 2D intervals
        dist_matrix = pairwise_distance(self.intervals_2d)
        self.assertEqual(dist_matrix.shape, (3, 3))

    def test_pairwise_distance_values(self):
        """Test that pairwise_distance computes correct values"""
        # Using Hausdorff distance
        dist_matrix = pairwise_distance(self.intervals_1d, metric="hausdorff")
        
        # Diagonal should be 0.0 (self-distance)
        self.assertEqual(dist_matrix[0, 0], 0.0)
        self.assertEqual(dist_matrix[1, 1], 0.0)
        self.assertEqual(dist_matrix[2, 2], 0.0)
        
        # Check specific pairs
        # Hausdorff for [1,3] and [2,4] = max(|2-1|, |4-3|) = max(1, 1) = 1
        self.assertEqual(dist_matrix[0, 1], 1.0)
        self.assertEqual(dist_matrix[1, 0], 1.0)
        
        # Hausdorff for [1,3] and [5,7] = max(|5-1|, |7-3|) = max(4, 4) = 4
        self.assertEqual(dist_matrix[0, 2], 4.0)
        self.assertEqual(dist_matrix[2, 0], 4.0)

    def test_pairwise_distance_metrics(self):
        """Test pairwise_distance with different metrics"""
        # Test with Euclidean distance
        dist_matrix_euc = pairwise_distance(self.intervals_1d, metric="euclidean")
        
        # Euclidean for [1,3] and [2,4] = sqrt((2-1)^2 + (4-3)^2) = sqrt(2)
        self.assertAlmostEqual(dist_matrix_euc[0, 1], np.sqrt(2), places=5)
        
        # Test with Manhattan distance
        dist_matrix_man = pairwise_distance(self.intervals_1d, metric="manhattan")
        
        # Manhattan for [1,3] and [2,4] = |2-1| + |4-3| = 1 + 1 = 2
        self.assertEqual(dist_matrix_man[0, 1], 2.0)

    def test_pairwise_distance_invalid_metric(self):
        """Test pairwise_distance with invalid metric"""
        with self.assertRaises(ValueError):
            pairwise_distance(self.intervals_1d, metric="invalid_metric")

    def test_cross_similarity_shape(self):
        """Test that cross_similarity returns correct shape"""
        # Cross similarity between sets of different sizes
        sim_matrix = cross_similarity(self.intervals_2d, self.intervals_b_2d)
        self.assertEqual(sim_matrix.shape, (3, 2))
        
        # Reverse order
        sim_matrix = cross_similarity(self.intervals_b_2d, self.intervals_2d)
        self.assertEqual(sim_matrix.shape, (2, 3))

    def test_cross_similarity_values(self):
        """Test that cross_similarity computes correct values"""
        sim_matrix = cross_similarity(self.intervals_2d, self.intervals_b_2d, metric="jaccard")
        
        # Check specific pairs
        # Intervals [1,3],[1,3] and [0,2],[0,2] have average Jaccard:
        # Dim 1: intersection [1,2] = 1, union [0,3] = 3, Jaccard = 1/3
        # Dim 2: intersection [1,2] = 1, union [0,3] = 3, Jaccard = 1/3
        # Average = 1/3
        self.assertAlmostEqual(sim_matrix[0, 0], 1/3, places=5)
        
        # Intervals [5,7],[5,7] and [0,2],[0,2] don't overlap
        # Jaccard = 0
        self.assertEqual(sim_matrix[2, 0], 0.0)

    def test_cross_distance_shape(self):
        """Test that cross_distance returns correct shape"""
        # Cross distance between sets of different sizes
        dist_matrix = cross_distance(self.intervals_2d, self.intervals_b_2d)
        self.assertEqual(dist_matrix.shape, (3, 2))
        
        # Reverse order
        dist_matrix = cross_distance(self.intervals_b_2d, self.intervals_2d)
        self.assertEqual(dist_matrix.shape, (2, 3))

    def test_cross_distance_values(self):
        """Test that cross_distance computes correct values"""
        dist_matrix = cross_distance(self.intervals_2d, self.intervals_b_2d, metric="hausdorff")
        
        # Check specific pairs
        # Intervals [1,3],[1,3] and [0,2],[0,2]:
        # Dim 1: max(|1-0|, |3-2|) = max(1, 1) = 1
        # Dim 2: max(|1-0|, |3-2|) = max(1, 1) = 1
        # Sum = 2
        self.assertEqual(dist_matrix[0, 0], 2.0)
        
        # Intervals [5,7],[5,7] and [3,5],[3,5]:
        # Dim 1: max(|5-3|, |7-5|) = max(2, 2) = 2
        # Dim 2: max(|5-3|, |7-5|) = max(2, 2) = 2
        # Sum = 4
        self.assertEqual(dist_matrix[2, 1], 4.0)

    def test_symmetry_in_pairwise(self):
        """Test that pairwise matrices are symmetric"""
        # Similarity matrix should be symmetric
        sim_matrix = pairwise_similarity(self.intervals_2d)
        self.assertTrue(np.allclose(sim_matrix, sim_matrix.T))
        
        # Distance matrix should be symmetric
        dist_matrix = pairwise_distance(self.intervals_2d)
        self.assertTrue(np.allclose(dist_matrix, dist_matrix.T))
    
    def test_cross_vs_pairwise(self):
        """Test that cross metrics match pairwise for identical sets"""
        # When both sets are the same, cross metrics should match pairwise
        cross_sim = cross_similarity(self.intervals_2d, self.intervals_2d)
        pairwise_sim = pairwise_similarity(self.intervals_2d)
        self.assertTrue(np.allclose(cross_sim, pairwise_sim))
        
        cross_dist = cross_distance(self.intervals_2d, self.intervals_2d)
        pairwise_dist = pairwise_distance(self.intervals_2d)
        self.assertTrue(np.allclose(cross_dist, pairwise_dist))

if __name__ == '__main__':
    unittest.main()
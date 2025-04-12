import unittest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from interClusLib.visualization.AbstractIntervalVisualization import AbstractIntervalVisualization

class TestAbstractIntervalVisualization(unittest.TestCase):
    """Test static and class methods of AbstractIntervalVisualization base class"""
    
    def setUp(self):
        """Set up test data for each test method"""
        # 2D test data with 3 intervals
        self.intervals_2d = np.array([
            [[0, 1], [0, 1]],  # First interval: x=[0,1], y=[0,1]
            [[1, 2], [1, 2]],  # Second interval: x=[1,2], y=[1,2]
            [[2, 3], [2, 3]]   # Third interval: x=[2,3], y=[2,3]
        ])
        
        # Centroid data
        self.centroids_2d = np.array([
            [[0.5, 1.5], [0.5, 1.5]],  # Centroid 1
            [[2.0, 3.0], [2.0, 3.0]]   # Centroid 2
        ])
        
        # Labels
        self.labels = np.array([0, 0, 1])  # First two in cluster 0, third in cluster 1
    
    def test_validate_intervals(self):
        """Test interval data validation"""
        # Correct data format should not raise errors
        AbstractIntervalVisualization.validate_intervals(self.intervals_2d)
        
        # Test dimension requirement
        AbstractIntervalVisualization.validate_intervals(self.intervals_2d, n_dims_required=2)
        
        # Test incorrect dimension requirement
        with self.assertRaises(ValueError):
            AbstractIntervalVisualization.validate_intervals(self.intervals_2d, n_dims_required=3)
        
        # Test incorrect data shape
        invalid_shape = np.array([[0, 1], [1, 2]])  # 2D array instead of 3D
        with self.assertRaises(ValueError):
            AbstractIntervalVisualization.validate_intervals(invalid_shape)
        
        # None value should pass validation
        AbstractIntervalVisualization.validate_intervals(None)
    
    def test_validate_centroids(self):
        """Test centroid data validation"""
        # Correct data format should not raise errors
        AbstractIntervalVisualization.validate_centroids(self.centroids_2d)
        
        # Test dimension requirement
        AbstractIntervalVisualization.validate_centroids(self.centroids_2d, n_dims_required=2)
        
        # Test incorrect dimension requirement
        with self.assertRaises(ValueError):
            AbstractIntervalVisualization.validate_centroids(self.centroids_2d, n_dims_required=3)
        
        # Test incorrect data shape
        invalid_shape = np.array([[0, 1], [1, 2]])  # 2D array instead of 3D
        with self.assertRaises(ValueError):
            AbstractIntervalVisualization.validate_centroids(invalid_shape)
        
        # None value should pass validation
        AbstractIntervalVisualization.validate_centroids(None)
    
    def test_setup_cluster_info(self):
        """Test cluster information setup"""
        # Using intervals and labels
        labels, unique_labels, n_clusters = AbstractIntervalVisualization.setup_cluster_info(
            self.intervals_2d, self.labels, None
        )
        self.assertTrue(np.array_equal(labels, self.labels))
        self.assertTrue(np.array_equal(unique_labels, np.array([0, 1])))
        self.assertEqual(n_clusters, 2)
        
        # Using only intervals (no labels)
        labels, unique_labels, n_clusters = AbstractIntervalVisualization.setup_cluster_info(
            self.intervals_2d, None, None
        )
        self.assertTrue(np.array_equal(labels, np.zeros(3, dtype=int)))
        self.assertTrue(np.array_equal(unique_labels, np.array([0])))
        self.assertEqual(n_clusters, 1)
        
        # Using only centroids
        labels, unique_labels, n_clusters = AbstractIntervalVisualization.setup_cluster_info(
            None, None, self.centroids_2d
        )
        self.assertEqual(labels.size, 0)
        self.assertTrue(np.array_equal(unique_labels, np.array([0, 1])))
        self.assertEqual(n_clusters, 2)
        
        # No intervals and centroids should raise error
        with self.assertRaises(ValueError):
            AbstractIntervalVisualization.setup_cluster_info(None, None, None)
    
    def test_generate_feature_names(self):
        """Test feature name generation"""
        # No names provided
        names = AbstractIntervalVisualization.generate_feature_names(3)
        self.assertEqual(names, ["Feature_1", "Feature_2", "Feature_3"])
        
        # Partial names provided
        names = AbstractIntervalVisualization.generate_feature_names(3, ["X", "Y"])
        self.assertEqual(names, ["X", "Y", "Feature_3"])
        
        # All names provided
        names = AbstractIntervalVisualization.generate_feature_names(2, ["X", "Y"])
        self.assertEqual(names, ["X", "Y"])
        
        # More names than needed
        names = AbstractIntervalVisualization.generate_feature_names(2, ["X", "Y", "Z"])
        self.assertEqual(names, ["X", "Y", "Z"])
        
        # Custom prefix
        names = AbstractIntervalVisualization.generate_feature_names(2, prefix="Dim_")
        self.assertEqual(names, ["Dim_1", "Dim_2"])
    
    def test_generate_cluster_colors(self):
        """Test cluster color generation"""
        # Generate colors for two clusters
        colors = AbstractIntervalVisualization.generate_cluster_colors(2)
        self.assertEqual(len(colors), 2)
        
        # Using different colormap
        colors = AbstractIntervalVisualization.generate_cluster_colors(3, cmap_name='viridis')
        self.assertEqual(len(colors), 3)
        
        # Verify color format (should be RGBA)
        self.assertEqual(len(colors[0]), 4)
    
    def test_get_feature_boundaries(self):
        """Test feature boundary calculation"""
        # Ensure test data is float type to avoid type conversion issues
        intervals_2d_float = self.intervals_2d.astype(float)
        centroids_2d_float = self.centroids_2d.astype(float)
        
        # Using only intervals - with default margin=0.1
        mins, maxs = AbstractIntervalVisualization.get_feature_boundaries(intervals_2d_float, None)
        # For range [0,3], margin 0.1 gives a margin value of 0.3
        np.testing.assert_allclose(mins, np.array([-0.3, -0.3]))  # Min should be [0,0] minus margin
        np.testing.assert_allclose(maxs, np.array([3.3, 3.3]))    # Max should be [3,3] plus margin
        
        # Using only centroids - with default margin=0.1
        mins, maxs = AbstractIntervalVisualization.get_feature_boundaries(None, centroids_2d_float)
        # For range [0.5,3], margin 0.1 gives a margin value of 0.25
        np.testing.assert_allclose(mins, np.array([0.25, 0.25]))  # Min should be [0.5,0.5] minus margin
        np.testing.assert_allclose(maxs, np.array([3.25, 3.25]))  # Max should be [3,3] plus margin
        
        # Using both intervals and centroids
        mins, maxs = AbstractIntervalVisualization.get_feature_boundaries(intervals_2d_float, centroids_2d_float)
        np.testing.assert_allclose(mins, np.array([-0.3, -0.3]))
        np.testing.assert_allclose(maxs, np.array([3.3, 3.3]))
        
        # Test custom margin
        mins, maxs = AbstractIntervalVisualization.get_feature_boundaries(intervals_2d_float, None, margin=0.2)
        # For range [0,3], margin 0.2 gives a margin value of 0.6
        np.testing.assert_allclose(mins, np.array([-0.6, -0.6]))
        np.testing.assert_allclose(maxs, np.array([3.6, 3.6]))
    
    def test_scale_to_unit(self):
        """Test scaling to unit range"""
        # Create feature min and max values
        feature_mins = np.array([0, 1])
        feature_maxs = np.array([10, 11])
        
        # Test scaling the first feature
        values = np.array([0, 5, 10])
        scaled = AbstractIntervalVisualization.scale_to_unit(values, 0, feature_mins, feature_maxs)
        np.testing.assert_allclose(scaled, np.array([0, 0.5, 1.0]))
        
        # Test scaling the second feature
        values = np.array([1, 6, 11])
        scaled = AbstractIntervalVisualization.scale_to_unit(values, 1, feature_mins, feature_maxs)
        np.testing.assert_allclose(scaled, np.array([0, 0.5, 1.0]))
        
        # Test case where min equals max
        feature_mins = np.array([5, 5])
        feature_maxs = np.array([5, 5])
        values = np.array([5, 5, 5])
        scaled = AbstractIntervalVisualization.scale_to_unit(values, 0, feature_mins, feature_maxs)
        np.testing.assert_allclose(scaled, np.array([0.5, 0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
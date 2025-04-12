import unittest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import matplotlib.patches as patches

# Import the classes to test
from interClusLib.visualization.AbstractIntervalVisualization import AbstractIntervalVisualization
from interClusLib.visualization.Interval2d import Interval2d


class TestInterval2d(unittest.TestCase):
    """Test suite for the Interval2d visualization class"""
    
    def setUp(self):
        """Set up test data for each test method"""
        # 2D test data with 3 intervals
        self.intervals_2d = np.array([
            [[0, 1], [0, 1]],  # First interval: x=[0,1], y=[0,1]
            [[1, 2], [1, 2]],  # Second interval: x=[1,2], y=[1,2]
            [[2, 3], [2, 3]]   # Third interval: x=[2,3], y=[2,3]
        ])
        
        # 1D test data (will be visualized as squares)
        self.intervals_1d = np.array([
            [[0, 1]],  # First interval: [0,1]
            [[1, 2]],  # Second interval: [1,2]
            [[2, 3]]   # Third interval: [2,3]
        ])
        
        # Higher dimension test data (should use only first 2 dimensions)
        self.intervals_3d = np.array([
            [[0, 1], [0, 1], [0, 1]],  # First interval
            [[1, 2], [1, 2], [1, 2]],  # Second interval
            [[2, 3], [2, 3], [2, 3]]   # Third interval
        ])
        
        # Centroids data
        self.centroids_2d = np.array([
            [[0.5, 1.5], [0.5, 1.5]],  # Centroid 1
            [[2.0, 3.0], [2.0, 3.0]]   # Centroid 2
        ])
        
        # Labels for the intervals
        self.labels = np.array([0, 0, 1])  # First two in cluster 0, third in cluster 1

    def test_init(self):
        """Test that the class can be instantiated"""
        # This class is primarily static methods, so just verify it can be imported
        self.assertIsNotNone(Interval2d)
    
    def test_process_intervals_2d(self):
        """Test processing of 2D intervals"""
        processed = Interval2d._process_intervals(self.intervals_2d)
        self.assertEqual(processed.shape, (3, 2, 2))
        np.testing.assert_array_equal(processed, self.intervals_2d)
    
    def test_process_intervals_1d(self):
        """Test processing of 1D intervals"""
        processed = Interval2d._process_intervals(self.intervals_1d)
        self.assertEqual(processed.shape, (3, 1, 2))
        np.testing.assert_array_equal(processed, self.intervals_1d)
    
    def test_process_intervals_higher_dim(self):
        """Test processing of intervals with more than 2 dimensions"""
        processed = Interval2d._process_intervals(self.intervals_3d)
        self.assertEqual(processed.shape, (3, 2, 2))  # Should take only first 2 dimensions
        np.testing.assert_array_equal(processed, self.intervals_3d[:, :2, :])
    
    def test_process_intervals_invalid_shape(self):
        """Test error is raised for invalid shape"""
        invalid_intervals = np.array([[0, 1], [1, 2]])  # 2D array instead of 3D
        with self.assertRaises(ValueError):
            Interval2d._process_intervals(invalid_intervals)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.patches.Rectangle')
    def test_draw_2d_intervals_no_labels(self, mock_rectangle, mock_subplots):
        """Test drawing intervals without labels"""
        # Setup mock
        mock_ax = MagicMock()
        mock_subplots.return_value = (MagicMock(), mock_ax)
        mock_rectangle_instance = MagicMock()
        mock_rectangle.return_value = mock_rectangle_instance
        
        # Call method
        legend_handles = Interval2d._draw_2d_intervals(mock_ax, self.intervals_2d)
        
        # Verify rectangles were created
        self.assertEqual(mock_rectangle.call_count, 3)
        mock_ax.add_patch.assert_called()
        
        # Verify legend handle was created
        self.assertEqual(len(legend_handles), 1)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.patches.Rectangle')
    def test_draw_2d_intervals_with_labels(self, mock_rectangle, mock_subplots):
        """Test drawing intervals with labels"""
        # Setup mock
        mock_ax = MagicMock()
        mock_subplots.return_value = (MagicMock(), mock_ax)
        mock_rectangle_instance = MagicMock()
        mock_rectangle.return_value = mock_rectangle_instance
        
        # Call method
        legend_handles = Interval2d._draw_2d_intervals(mock_ax, self.intervals_2d, self.labels)
        
        # Verify rectangles were created
        self.assertEqual(mock_rectangle.call_count, 3)
        mock_ax.add_patch.assert_called()
        
        # Verify legend handles were created (one for each cluster)
        self.assertEqual(len(legend_handles), 2)
    
    @patch('matplotlib.pyplot.subplots')
    def test_draw_2d_intervals_max_samples(self, mock_subplots):
        """Test drawing intervals with max_samples_per_cluster limit"""
        # Setup mock
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Call method with max_samples_per_cluster=1
        Interval2d._draw_2d_intervals(mock_ax, self.intervals_2d, self.labels, max_samples_per_cluster=1)
        
        # Verify only 2 rectangles were added (1 per cluster)
        self.assertEqual(mock_ax.add_patch.call_count, 2)
    
    @patch('matplotlib.pyplot.subplots')
    def test_visualize_basic(self, mock_subplots):
        """Test basic visualization function"""
        # Setup mock
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Call method
        fig, ax = Interval2d.visualize(intervals=self.intervals_2d)
        
        # Verify figure was created and configured
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
        mock_ax.set_xlabel.assert_called()
        mock_ax.set_ylabel.assert_called()
        mock_ax.set_title.assert_called()
        mock_ax.set_xlim.assert_called()
        mock_ax.set_ylim.assert_called()
    
    @patch('matplotlib.pyplot.subplots')
    def test_visualize_with_centroids(self, mock_subplots):
        """Test visualization with centroids"""
        # Setup mock
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Call method
        fig, ax = Interval2d.visualize(
            intervals=self.intervals_2d,
            centroids=self.centroids_2d,
            labels=self.labels
        )
        
        # Verify figure was created and configured
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
        # Legend should be added when centroids are included
        mock_ax.legend.assert_called()
    
    @patch('matplotlib.pyplot.subplots')
    def test_visualize_only_centroids(self, mock_subplots):
        """Test visualization with only centroids"""
        # Setup mock
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Call method
        fig, ax = Interval2d.visualize(centroids=self.centroids_2d)
        
        # Verify figure was created and configured
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
    
    def test_visualize_empty(self):
        """Test visualize raises error when no data provided"""
        with self.assertRaises(ValueError):
            Interval2d.visualize()
    
    @patch('matplotlib.pyplot.subplots')
    def test_visualize_feature_names(self, mock_subplots):
        """Test visualization with custom feature names"""
        # Setup mock
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Call method with feature names
        feature_names = ["Feature 1", "Feature 2"]
        fig, ax = Interval2d.visualize(
            intervals=self.intervals_2d,
            feature_names=feature_names
        )
        
        # Verify axis labels were set correctly
        mock_ax.set_xlabel.assert_called_with(feature_names[0])
        mock_ax.set_ylabel.assert_called_with(feature_names[1])
    
    @patch('matplotlib.pyplot.subplots')
    def test_visualize_1d_intervals(self, mock_subplots):
        """Test visualization of 1D intervals"""
        # Setup mock
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Call method with 1D intervals
        fig, ax = Interval2d.visualize(intervals=self.intervals_1d)
        
        # Verify equal aspect ratio is set for 1D visualization
        mock_ax.set_aspect.assert_called_with('equal')
    
    @patch('matplotlib.pyplot.subplots')
    def test_visualize_filled_intervals(self, mock_subplots):
        """Test visualization with filled intervals"""
        # Setup mock
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Call method with fill_intervals=True
        fig, ax = Interval2d.visualize(
            intervals=self.intervals_2d,
            fill_intervals=True
        )
        
        # We can't easily verify the fill parameter without mocking Rectangle,
        # but we can ensure the method runs without errors
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)


if __name__ == "__main__":
    unittest.main()
import unittest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import matplotlib.patches as mpatches
import warnings

# Import the classes to test
from interClusLib.visualization.AbstractIntervalVisualization import AbstractIntervalVisualization
from interClusLib.visualization.IntervalParallelCoordinates import IntervalParallelCoordinates


class TestIntervalParallelCoordinates(unittest.TestCase):
    """Test suite for the IntervalParallelCoordinates visualization class"""
    
    def setUp(self):
        """Set up test data for each test method"""
        # Filter MatplotlibDeprecationWarning
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
        
        # Create test data with 3 intervals, 4 features
        self.intervals = np.array([
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],  # First interval
            [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],  # Second interval
            [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]   # Third interval
        ]).astype(float)
        
        # Centroids data (2 clusters, 4 features)
        self.centroids = np.array([
            [[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.5, 1.5]],  # Centroid 1
            [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]   # Centroid 2
        ]).astype(float)
        
        # Labels for the intervals
        self.labels = np.array([0, 0, 1])  # First two in cluster 0, third in cluster 1

    def tearDown(self):
        """Clean up after each test"""
        # Reset warnings filter
        warnings.resetwarnings()
    
    def test_init(self):
        """Test that the class can be instantiated"""
        # This class is primarily static methods, so just verify it can be imported
        self.assertIsNotNone(IntervalParallelCoordinates)
    
    def test_generate_bezier_curve(self):
        """Test bezier curve generation"""
        # Generate a simple bezier curve
        p0 = np.array([0.0, 0.0])
        p1 = np.array([0.33, 0.0])
        p2 = np.array([0.66, 1.0])
        p3 = np.array([1.0, 1.0])
        
        curve = IntervalParallelCoordinates._generate_bezier_curve(p0, p1, p2, p3, num_points=10)
        
        # Verify shape and endpoints
        self.assertEqual(curve.shape, (10, 2))
        np.testing.assert_almost_equal(curve[0], p0, decimal=6)
        np.testing.assert_almost_equal(curve[-1], p3, decimal=6)
        
        # Verify some properties of the curve
        x_values = curve[:, 0]
        y_values = curve[:, 1]
        
        # X values should be monotonically increasing
        self.assertTrue(np.all(np.diff(x_values) > 0))
        
        # All points should be within the convex hull of the control points
        for point in curve:
            x, y = point
            self.assertTrue(0 <= x <= 1)
            self.assertTrue(0 <= y <= 1)
    
    def test_compute_control_points(self):
        """Test computation of control points for bezier curves"""
        # Test with simple inputs
        p_left = np.array([0.0, 0.5])
        p_right = np.array([1.0, 0.5])
        
        # Without bundling (no centroid)
        (mid_x, mid_y), (cp1_x, cp1_y), (cp2_x, cp2_y), (cp3_x, cp3_y), (cp4_x, cp4_y) = \
            IntervalParallelCoordinates._compute_control_points(p_left, p_right, alpha=1/6, beta=0.8, centroid=None)
        
        # Test midpoint
        self.assertAlmostEqual(mid_x, 0.5)
        self.assertAlmostEqual(mid_y, 0.5)
        
        # Test control points for left segment
        self.assertAlmostEqual(cp1_x, p_left[0] + (1/6) * (mid_x - p_left[0]))
        self.assertAlmostEqual(cp1_y, p_left[1])
        
        self.assertAlmostEqual(cp2_x, mid_x - (1/6) * (mid_x - p_left[0]))
        self.assertAlmostEqual(cp2_y, mid_y)
        
        # Test control points for right segment
        self.assertAlmostEqual(cp3_x, mid_x + (1/6) * (p_right[0] - mid_x))
        self.assertAlmostEqual(cp3_y, mid_y)
        
        self.assertAlmostEqual(cp4_x, p_right[0] - (1/6) * (p_right[0] - mid_x))
        self.assertAlmostEqual(cp4_y, p_right[1])
        
        # With bundling (with centroid)
        centroid = np.array([0.5, 0.8])  # higher than midpoint
        (mid_x, mid_y), (cp1_x, cp1_y), (cp2_x, cp2_y), (cp3_x, cp3_y), (cp4_x, cp4_y) = \
            IntervalParallelCoordinates._compute_control_points(p_left, p_right, alpha=1/6, beta=0.8, centroid=centroid)
        
        # Midpoint should be adjusted toward centroid
        self.assertAlmostEqual(mid_x, 0.5)  # x doesn't change
        self.assertTrue(mid_y > 0.5)  # y should be pulled up toward 0.8
        self.assertTrue(mid_y < 0.8)  # but not all the way (beta < 1)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.patches.Polygon')
    def test_visualize_basic(self, mock_polygon, mock_subplots):
        """Test basic visualization function"""
        # Setup mocks
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_polygon_instance = MagicMock()
        mock_polygon.return_value = mock_polygon_instance
        
        # Call the method
        fig, ax = IntervalParallelCoordinates.visualize(intervals=self.intervals)
        
        # Verify figure was created and configured
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
        mock_ax.set_xlim.assert_called()
        mock_ax.set_ylim.assert_called()
        mock_ax.set_xticks.assert_called_with([])
        mock_ax.set_yticks.assert_called_with([])
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.patches.Polygon')
    def test_visualize_with_labels(self, mock_polygon, mock_subplots):
        """Test visualization with labels"""
        # Setup mocks
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_polygon_instance = MagicMock()
        mock_polygon.return_value = mock_polygon_instance
        
        # Call the method
        fig, ax = IntervalParallelCoordinates.visualize(
            intervals=self.intervals,
            labels=self.labels
        )
        
        # Verify figure was created and configured with legend
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
        mock_ax.legend.assert_called()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.patches.Polygon')
    def test_visualize_with_centroids(self, mock_polygon, mock_subplots):
        """Test visualization with centroids"""
        # Setup mocks
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_polygon_instance = MagicMock()
        mock_polygon.return_value = mock_polygon_instance
        
        # Call the method
        fig, ax = IntervalParallelCoordinates.visualize(
            intervals=self.intervals,
            centroids=self.centroids,
            labels=self.labels
        )
        
        # Verify figure was created and configured with legend
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
        mock_ax.legend.assert_called()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.patches.Polygon')
    def test_visualize_only_centroids(self, mock_polygon, mock_subplots):
        """Test visualization with only centroids"""
        # Setup mocks
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_polygon_instance = MagicMock()
        mock_polygon.return_value = mock_polygon_instance
        
        # Call the method
        fig, ax = IntervalParallelCoordinates.visualize(centroids=self.centroids)
        
        # Verify figure was created and configured
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.patches.Polygon')
    def test_visualize_custom_features(self, mock_polygon, mock_subplots):
        """Test visualization with custom feature names and other parameters"""
        # Setup mocks
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_polygon_instance = MagicMock()
        mock_polygon.return_value = mock_polygon_instance
        
        # Custom feature names
        feature_names = ["Dim 1", "Dim 2", "Dim 3", "Dim 4"]
        
        # Call the method with custom parameters
        fig, ax = IntervalParallelCoordinates.visualize(
            intervals=self.intervals,
            centroids=self.centroids,
            labels=self.labels,
            feature_names=feature_names,
            alpha=0.2,
            beta=0.5,
            uncertainty_alpha=0.3,
            centroid_alpha=0.7,
            use_bundling=False,
            title="Custom Title"
        )
        
        # Verify figure was created and configured
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
        mock_ax.set_title.assert_called_with("Custom Title")
    
    def test_visualize_empty(self):
        """Test visualize raises error when no data provided"""
        with self.assertRaises(ValueError):
            IntervalParallelCoordinates.visualize()
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.patches.Polygon')
    def test_max_samples_per_cluster(self, mock_polygon, mock_subplots):
        """Test the max_samples_per_cluster parameter"""
        # Setup mocks
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_polygon_instance = MagicMock()
        mock_polygon.return_value = mock_polygon_instance
        
        # Create more test data
        intervals_large = np.tile(self.intervals, (5, 1, 1))  # 15 intervals
        labels_large = np.array([0, 0, 1] * 5)  # 10 in cluster 0, 5 in cluster 1
        
        # Call with max_samples_per_cluster=2
        IntervalParallelCoordinates.visualize(
            intervals=intervals_large,
            labels=labels_large,
            max_samples_per_cluster=2
        )
        
        # With max_samples_per_cluster=2, we should have at most 2 samples per cluster
        # But verifying this with mocks is difficult because most of the drawing
        # happens inside the method. We're primarily checking that it runs without error.
    
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.patches.Polygon')
    def test_bundling_parameter(self, mock_polygon, mock_subplots):
        """Test the bundling parameters"""
        # Setup mocks
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_polygon_instance = MagicMock()
        mock_polygon.return_value = mock_polygon_instance
        
        # Test with bundling turned off
        IntervalParallelCoordinates.visualize(
            intervals=self.intervals,
            use_bundling=False
        )
        
        # Test with different bundling strength
        IntervalParallelCoordinates.visualize(
            intervals=self.intervals,
            beta=0.3  # lower bundling strength
        )
        
        # Test with different curve smoothness
        IntervalParallelCoordinates.visualize(
            intervals=self.intervals,
            alpha=0.25  # maximum smoothness
        )
        
        # Again, we're primarily checking these run without error


if __name__ == "__main__":
    unittest.main()
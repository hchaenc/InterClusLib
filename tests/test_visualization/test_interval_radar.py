import unittest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import matplotlib.patches as mpatches
import warnings

# Import the classes to test
from interClusLib.visualization.AbstractIntervalVisualization import AbstractIntervalVisualization
from interClusLib.visualization.IntervalRadarChart import IntervalRadarChart


class TestIntervalRadarChart(unittest.TestCase):
    """Test suite for the IntervalRadarChart visualization class"""
    
    def setUp(self):
        """Set up test data for each test method"""
        # Filter MatplotlibDeprecationWarning
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
        
        # Create test data with 3 intervals, 5 features
        self.intervals = np.array([
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],  # First interval
            [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],  # Second interval
            [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]   # Third interval
        ]).astype(float)
        
        # Centroids data (2 clusters, 5 features)
        self.centroids = np.array([
            [[0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.5, 1.5], [0.5, 1.5]],  # Centroid 1
            [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]   # Centroid 2
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
        self.assertIsNotNone(IntervalRadarChart)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplots_adjust')
    def test_visualize_basic(self, mock_subplots_adjust, mock_figure):
        """Test basic visualization function"""
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Call the method
        fig, ax = IntervalRadarChart.visualize(intervals=self.intervals)
        
        # Verify figure was created and configured
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
        mock_ax.set_ylim.assert_called_with(0, 1)
        mock_ax.set_xticks.assert_called_with([])
        mock_ax.set_yticks.assert_called_with([])
        mock_subplots_adjust.assert_called()
    
    @patch('matplotlib.pyplot.figure')
    def test_visualize_with_labels(self, mock_figure):
        """Test visualization with labels"""
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Call the method
        fig, ax = IntervalRadarChart.visualize(
            intervals=self.intervals,
            labels=self.labels
        )
        
        # Verify figure was created and configured with legend
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
        mock_ax.legend.assert_called()
    
    @patch('matplotlib.pyplot.figure')
    def test_visualize_with_centroids(self, mock_figure):
        """Test visualization with centroids"""
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Call the method
        fig, ax = IntervalRadarChart.visualize(
            intervals=self.intervals,
            centroids=self.centroids,
            labels=self.labels
        )
        
        # Verify figure was created and configured with legend
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
        mock_ax.legend.assert_called()
    
    @patch('matplotlib.pyplot.figure')
    def test_visualize_only_centroids(self, mock_figure):
        """Test visualization with only centroids"""
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Call the method
        fig, ax = IntervalRadarChart.visualize(centroids=self.centroids)
        
        # Verify figure was created and configured
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
    
    @patch('matplotlib.pyplot.figure')
    def test_visualize_custom_features(self, mock_figure):
        """Test visualization with custom feature names and other parameters"""
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Custom feature names
        feature_names = ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"]
        
        # Call the method with custom parameters
        fig, ax = IntervalRadarChart.visualize(
            intervals=self.intervals,
            centroids=self.centroids,
            labels=self.labels,
            feature_names=feature_names,
            alpha=0.3,
            centroid_alpha=0.6,
            title="Custom Title"
        )
        
        # Verify figure was created and configured
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
        mock_fig.suptitle.assert_called_with("Custom Title", y=0.98, fontsize=18, fontweight='bold')
    
    def test_visualize_empty(self):
        """Test visualize raises error when no data provided"""
        with self.assertRaises(ValueError):
            IntervalRadarChart.visualize()
    
    @patch('matplotlib.pyplot.figure')
    def test_max_samples_per_cluster(self, mock_figure):
        """Test the max_samples_per_cluster parameter"""
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Create more test data
        intervals_large = np.tile(self.intervals, (5, 1, 1))  # 15 intervals
        labels_large = np.array([0, 0, 1] * 5)  # 10 in cluster 0, 5 in cluster 1
        
        # Call with max_samples_per_cluster=2
        IntervalRadarChart.visualize(
            intervals=intervals_large,
            labels=labels_large,
            max_samples_per_cluster=2
        )
        
        # With max_samples_per_cluster=2, we should have at most 2 samples per cluster
        # But verifying this with mocks is difficult because most of the drawing
        # happens inside the method. We're primarily checking that it runs without error.
    
    @patch('matplotlib.pyplot.figure')
    def test_margin_parameter(self, mock_figure):
        """Test the margin parameter"""
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Call the method with a custom margin
        with patch.object(AbstractIntervalVisualization, 'get_feature_boundaries') as mock_get_boundaries:
            # Set up the mock to return known values
            mock_get_boundaries.return_value = (np.zeros(5), np.ones(5))
            
            IntervalRadarChart.visualize(
                intervals=self.intervals,
                margin=0.2  # Custom margin
            )
            
            # Verify the margin was passed correctly
            mock_get_boundaries.assert_called_with(self.intervals, None, 0.2)
    
    @patch('matplotlib.pyplot.figure')
    def test_feature_boundaries_calculation(self, mock_figure):
        """Test that feature boundaries are calculated correctly"""
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Test with both intervals and centroids to ensure proper boundary calculation
        with patch.object(AbstractIntervalVisualization, 'get_feature_boundaries') as mock_get_boundaries:
            # Setup the mock to return fixed boundaries
            mins = np.zeros(5)
            maxs = np.ones(5) * 3.0
            mock_get_boundaries.return_value = (mins, maxs)
            
            # Call the visualization method
            IntervalRadarChart.visualize(
                intervals=self.intervals,
                centroids=self.centroids
            )
            
            # Verify the method was called with both datasets
            mock_get_boundaries.assert_called_with(self.intervals, self.centroids, 0.1)
    
    @patch('matplotlib.pyplot.figure')
    def test_scaling_to_unit_range(self, mock_figure):
        """Test that values are scaled correctly to the unit range"""
        # Setup mocks
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Patch both get_feature_boundaries and scale_to_unit to test their interaction
        with patch.object(AbstractIntervalVisualization, 'get_feature_boundaries') as mock_get_boundaries:
            with patch.object(AbstractIntervalVisualization, 'scale_to_unit') as mock_scale_to_unit:
                # Setup the mocks
                mins = np.zeros(5)
                maxs = np.ones(5) * 3.0
                mock_get_boundaries.return_value = (mins, maxs)
                mock_scale_to_unit.return_value = 0.5  # Middle of the unit range
                
                # Call the visualization method
                IntervalRadarChart.visualize(intervals=self.intervals)
                
                # Verify scale_to_unit was called with correct parameters
                # It should be called for each feature of each interval
                # The exact count depends on implementation details, so we just check it was called
                mock_scale_to_unit.assert_called()


if __name__ == "__main__":
    unittest.main()
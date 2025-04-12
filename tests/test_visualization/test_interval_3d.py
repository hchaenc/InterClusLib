import unittest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings

# Import the classes to test
from interClusLib.visualization.AbstractIntervalVisualization import AbstractIntervalVisualization
from interClusLib.visualization.Interval3d import Interval3d


class TestInterval3d(unittest.TestCase):
    """Test suite for the Interval3d visualization class"""
    
    def setUp(self):
        """Set up test data for each test method"""
        # Filter MatplotlibDeprecationWarning
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
        
        # 3D test data with 3 intervals
        self.intervals_3d = np.array([
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],  # First interval: x=[0,1], y=[0,1], z=[0,1]
            [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],  # Second interval: x=[1,2], y=[1,2], z=[1,2]
            [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]   # Third interval: x=[2,3], y=[2,3], z=[2,3]
        ])
        
        # Higher dimension test data (should use only first 3 dimensions)
        self.intervals_4d = np.array([
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],  # First interval
            [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],  # Second interval
            [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]   # Third interval
        ])
        
        # 2D test data (should raise ValueError for 3D visualization)
        self.intervals_2d = np.array([
            [[0.0, 1.0], [0.0, 1.0]],  # First interval: x=[0,1], y=[0,1]
            [[1.0, 2.0], [1.0, 2.0]],  # Second interval: x=[1,2], y=[1,2]
            [[2.0, 3.0], [2.0, 3.0]]   # Third interval: x=[2,3], y=[2,3]
        ])
        
        # Centroids data
        self.centroids_3d = np.array([
            [[0.5, 1.5], [0.5, 1.5], [0.5, 1.5]],  # Centroid 1
            [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]   # Centroid 2
        ])
        
        # Labels for the intervals
        self.labels = np.array([0, 0, 1])  # First two in cluster 0, third in cluster 1

    def tearDown(self):
        """Clean up after each test"""
        # Reset warnings filter
        warnings.resetwarnings()
    
    def test_init(self):
        """Test that the class can be instantiated"""
        # This class is primarily static methods, so just verify it can be imported
        self.assertIsNotNone(Interval3d)
    
    def test_process_intervals_3d(self):
        """Test processing of 3D intervals"""
        processed = Interval3d._process_intervals(self.intervals_3d)
        self.assertEqual(processed.shape, (3, 3, 2))
        np.testing.assert_array_equal(processed, self.intervals_3d)
    
    def test_process_intervals_higher_dim(self):
        """Test processing of intervals with more than 3 dimensions"""
        processed = Interval3d._process_intervals(self.intervals_4d)
        self.assertEqual(processed.shape, (3, 3, 2))  # Should take only first 3 dimensions
        np.testing.assert_array_equal(processed, self.intervals_4d[:, :3, :])
    
    def test_process_intervals_lower_dim(self):
        """Test error is raised for dimensions less than 3"""
        with self.assertRaises(ValueError):
            Interval3d._process_intervals(self.intervals_2d)
    
    def test_get_cube_faces(self):
        """Test generation of cube faces from coordinates"""
        # Test cube [0,1] × [0,1] × [0,1]
        faces = Interval3d._get_cube_faces(0, 1, 0, 1, 0, 1)
        
        # Verify we have 6 faces
        self.assertEqual(len(faces), 6)
        
        # Each face should have 4 corners
        for face in faces:
            self.assertEqual(len(face), 4)
        
        # Verify first face (x_lower face) has correct coordinates
        # Should have corners at (0,0,0), (0,1,0), (0,1,1), (0,0,1)
        x_lower_face = faces[0]
        self.assertEqual(x_lower_face[0], (0, 0, 0))  # c000
        self.assertEqual(x_lower_face[1], (0, 1, 0))  # c010
        self.assertEqual(x_lower_face[2], (0, 1, 1))  # c011
        self.assertEqual(x_lower_face[3], (0, 0, 1))  # c001
    
    @patch('mpl_toolkits.mplot3d.art3d.Poly3DCollection')
    def test_draw_3d_intervals_no_labels(self, mock_poly3d):
        """Test drawing intervals without labels"""
        # Setup mock
        mock_ax = MagicMock()
        mock_poly3d_instance = MagicMock()
        mock_poly3d.return_value = mock_poly3d_instance
        
        # Call method
        with patch.object(Interval3d, '_get_cube_faces', return_value=[[(0,0,0), (0,1,0), (0,1,1), (0,0,1)]]):
            legend_handles = Interval3d._draw_3d_intervals(mock_ax, self.intervals_3d)
        
        # Verify add_collection3d was called for each interval
        self.assertEqual(mock_ax.add_collection3d.call_count, 3)
        
        # Verify legend handle was created
        self.assertEqual(len(legend_handles), 1)
    
    @patch('mpl_toolkits.mplot3d.art3d.Poly3DCollection')
    def test_draw_3d_intervals_with_labels(self, mock_poly3d):
        """Test drawing intervals with labels"""
        # Setup mock
        mock_ax = MagicMock()
        mock_poly3d_instance = MagicMock()
        mock_poly3d.return_value = mock_poly3d_instance
        
        # Call method
        with patch.object(Interval3d, '_get_cube_faces', return_value=[[(0,0,0), (0,1,0), (0,1,1), (0,0,1)]]):
            legend_handles = Interval3d._draw_3d_intervals(mock_ax, self.intervals_3d, self.labels)
        
        # Verify add_collection3d was called for each interval
        self.assertEqual(mock_ax.add_collection3d.call_count, 3)
        
        # Verify legend handles were created (one for each cluster)
        self.assertEqual(len(legend_handles), 2)
    
    @patch('matplotlib.pyplot.figure')
    @patch('mpl_toolkits.mplot3d.art3d.Poly3DCollection')
    def test_draw_3d_intervals_max_samples(self, mock_poly3d, mock_figure):
        """Test drawing intervals with max_samples_per_cluster limit"""
        # Setup mock
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        mock_poly3d_instance = MagicMock()
        mock_poly3d.return_value = mock_poly3d_instance
        
        # Call method with max_samples_per_cluster=1
        Interval3d._draw_3d_intervals(mock_ax, self.intervals_3d, self.labels, max_samples_per_cluster=1)
        
        # Verify only 2 cuboids were added (1 per cluster)
        self.assertEqual(mock_ax.add_collection3d.call_count, 2)
    
    @patch('matplotlib.pyplot.figure')
    def test_visualize_basic(self, mock_figure):
        """Test basic visualization function"""
        # Setup mock
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Call method
        fig, ax = Interval3d.visualize(intervals=self.intervals_3d)
        
        # Verify figure was created and configured
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
        mock_ax.set_xlabel.assert_called()
        mock_ax.set_ylabel.assert_called()
        mock_ax.set_zlabel.assert_called()
        mock_ax.set_title.assert_called()
        mock_ax.set_xlim.assert_called()
        mock_ax.set_ylim.assert_called()
        mock_ax.set_zlim.assert_called()
    
    @patch('matplotlib.pyplot.figure')
    def test_visualize_with_centroids(self, mock_figure):
        """Test visualization with centroids"""
        # Setup mock
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Call method
        fig, ax = Interval3d.visualize(
            intervals=self.intervals_3d,
            centroids=self.centroids_3d,
            labels=self.labels
        )
        
        # Verify figure was created and configured
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
        # Legend should be added when centroids are included
        mock_ax.legend.assert_called()
    
    @patch('matplotlib.pyplot.figure')
    def test_visualize_only_centroids(self, mock_figure):
        """Test visualization with only centroids"""
        # Setup mock
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Call method
        fig, ax = Interval3d.visualize(centroids=self.centroids_3d)
        
        # Verify figure was created and configured
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
    
    def test_visualize_empty(self):
        """Test visualize raises error when no data provided"""
        with self.assertRaises(ValueError):
            Interval3d.visualize()
    
    @patch('matplotlib.pyplot.figure')
    def test_visualize_feature_names(self, mock_figure):
        """Test visualization with custom feature names"""
        # Setup mock
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax
        
        # Call method with feature names
        feature_names = ["Feature 1", "Feature 2", "Feature 3"]
        fig, ax = Interval3d.visualize(
            intervals=self.intervals_3d,
            feature_names=feature_names
        )
        
        # Verify axis labels were set correctly
        mock_ax.set_xlabel.assert_called_with(feature_names[0])
        mock_ax.set_ylabel.assert_called_with(feature_names[1])
        mock_ax.set_zlabel.assert_called_with(feature_names[2])
    
    @patch('matplotlib.pyplot.figure')
    def test_visualize_invalid_dimensions(self, mock_figure):
        """Test visualization with insufficient dimensions"""
        # Should raise ValueError for 2D intervals
        with self.assertRaises(ValueError):
            Interval3d.visualize(intervals=self.intervals_2d)
    
    def test_invalid_input_dimensions(self):
        """Test error is raised for invalid input dimensions"""
        with self.assertRaises(ValueError):
            Interval3d._draw_3d_intervals(MagicMock(), self.intervals_2d)


if __name__ == "__main__":
    unittest.main()
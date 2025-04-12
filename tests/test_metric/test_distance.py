import unittest
import numpy as np
from interClusLib.metric.distance import (
    hausdorff_distance, 
    euclidean_distance, 
    manhattan_distance,
    DISTANCE_FUNCTIONS
)

class TestDistanceFunctions(unittest.TestCase):
    """Test suite for interval distance functions"""
    
    def setUp(self):
        """Set up test data"""
        # Single-dimensional intervals
        self.interval1_1d = np.array([1.0, 5.0])
        self.interval2_1d = np.array([2.0, 7.0])
        self.interval3_1d = np.array([1.0, 5.0])  # Same as interval1_1d
        
        # Multi-dimensional intervals
        self.interval1_2d = np.array([[1.0, 5.0], [2.0, 6.0]])  # 2D interval
        self.interval2_2d = np.array([[2.0, 7.0], [3.0, 8.0]])  # 2D interval
        self.interval3_2d = np.array([[1.0, 5.0], [2.0, 6.0]])  # Same as interval1_2d
        
        # Higher-dimensional intervals
        self.interval1_3d = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])  # 3D interval
        self.interval2_3d = np.array([[2.0, 7.0], [3.0, 8.0], [4.0, 9.0]])  # 3D interval

    def test_hausdorff_distance_1d(self):
        """Test Hausdorff distance for 1D intervals"""
        # Distance between different intervals
        distance = hausdorff_distance(self.interval1_1d, self.interval2_1d)
        self.assertEqual(distance, 2.0)  # max(|2-1|, |7-5|) = max(1, 2) = 2
        
        # Distance between same intervals should be 0
        self.assertEqual(hausdorff_distance(self.interval1_1d, self.interval3_1d), 0.0)
        
        # Distance should be symmetric
        self.assertEqual(
            hausdorff_distance(self.interval1_1d, self.interval2_1d),
            hausdorff_distance(self.interval2_1d, self.interval1_1d)
        )

    def test_hausdorff_distance_2d(self):
        """Test Hausdorff distance for 2D intervals"""
        # Distance between different intervals
        distance = hausdorff_distance(self.interval1_2d, self.interval2_2d)
        # For 2D: sum of max(|2-1|, |7-5|) and max(|3-2|, |8-6|) = 2 + 2 = 4
        self.assertEqual(distance, 4.0)
        
        # Distance between same intervals should be 0
        self.assertEqual(hausdorff_distance(self.interval1_2d, self.interval3_2d), 0.0)
        
        # Distance should be symmetric
        self.assertEqual(
            hausdorff_distance(self.interval1_2d, self.interval2_2d),
            hausdorff_distance(self.interval2_2d, self.interval1_2d)
        )

    def test_hausdorff_distance_3d(self):
        """Test Hausdorff distance for 3D intervals"""
        # Distance between different intervals
        distance = hausdorff_distance(self.interval1_3d, self.interval2_3d)
        # For 3D: sum of max distances for each dimension
        self.assertEqual(distance, 6.0)  # 2 + 2 + 2 = 6
        
        # Distance should be symmetric
        self.assertEqual(
            hausdorff_distance(self.interval1_3d, self.interval2_3d),
            hausdorff_distance(self.interval2_3d, self.interval1_3d)
        )

    def test_euclidean_distance_1d(self):
        """Test Euclidean distance for 1D intervals"""
        # Distance between different intervals
        distance = euclidean_distance(self.interval1_1d, self.interval2_1d)
        # sqrt((2-1)^2 + (7-5)^2) = sqrt(1 + 4) = sqrt(5)
        self.assertAlmostEqual(distance, np.sqrt(5))
        
        # Distance between same intervals should be 0
        self.assertEqual(euclidean_distance(self.interval1_1d, self.interval3_1d), 0.0)
        
        # Distance should be symmetric
        self.assertAlmostEqual(
            euclidean_distance(self.interval1_1d, self.interval2_1d),
            euclidean_distance(self.interval2_1d, self.interval1_1d)
        )

    def test_euclidean_distance_2d(self):
        """Test Euclidean distance for 2D intervals"""
        # Distance between different intervals
        distance = euclidean_distance(self.interval1_2d, self.interval2_2d)
        # sqrt(((2-1)^2 + (7-5)^2) + ((3-2)^2 + (8-6)^2)) = sqrt(5 + 5) = sqrt(10)
        self.assertAlmostEqual(distance, np.sqrt(10))
        
        # Distance between same intervals should be 0
        self.assertEqual(euclidean_distance(self.interval1_2d, self.interval3_2d), 0.0)
        
        # Distance should be symmetric
        self.assertAlmostEqual(
            euclidean_distance(self.interval1_2d, self.interval2_2d),
            euclidean_distance(self.interval2_2d, self.interval1_2d)
        )

    def test_euclidean_distance_3d(self):
        """Test Euclidean distance for 3D intervals"""
        # Distance between different intervals
        distance = euclidean_distance(self.interval1_3d, self.interval2_3d)
        # sqrt(5 + 5 + 5) = sqrt(15)
        self.assertAlmostEqual(distance, np.sqrt(15))
        
        # Distance should be symmetric
        self.assertAlmostEqual(
            euclidean_distance(self.interval1_3d, self.interval2_3d),
            euclidean_distance(self.interval2_3d, self.interval1_3d)
        )

    def test_manhattan_distance_1d(self):
        """Test Manhattan distance for 1D intervals"""
        # Distance between different intervals
        distance = manhattan_distance(self.interval1_1d, self.interval2_1d)
        # |2-1| + |7-5| = 1 + 2 = 3
        self.assertEqual(distance, 3.0)
        
        # Distance between same intervals should be 0
        self.assertEqual(manhattan_distance(self.interval1_1d, self.interval3_1d), 0.0)
        
        # Distance should be symmetric
        self.assertEqual(
            manhattan_distance(self.interval1_1d, self.interval2_1d),
            manhattan_distance(self.interval2_1d, self.interval1_1d)
        )

    def test_manhattan_distance_2d(self):
        """Test Manhattan distance for 2D intervals"""
        # Distance between different intervals
        distance = manhattan_distance(self.interval1_2d, self.interval2_2d)
        # (|2-1| + |7-5|) + (|3-2| + |8-6|) = 3 + 3 = 6
        self.assertEqual(distance, 6.0)
        
        # Distance between same intervals should be 0
        self.assertEqual(manhattan_distance(self.interval1_2d, self.interval3_2d), 0.0)
        
        # Distance should be symmetric
        self.assertEqual(
            manhattan_distance(self.interval1_2d, self.interval2_2d),
            manhattan_distance(self.interval2_2d, self.interval1_2d)
        )

    def test_manhattan_distance_3d(self):
        """Test Manhattan distance for 3D intervals"""
        # Distance between different intervals
        distance = manhattan_distance(self.interval1_3d, self.interval2_3d)
        # 3 + 3 + 3 = 9
        self.assertEqual(distance, 9.0)
        
        # Distance should be symmetric
        self.assertEqual(
            manhattan_distance(self.interval1_3d, self.interval2_3d),
            manhattan_distance(self.interval2_3d, self.interval1_3d)
        )

    def test_distance_functions_dictionary(self):
        """Test the DISTANCE_FUNCTIONS dictionary"""
        # Check that all defined functions are in the dictionary
        self.assertIn("hausdorff", DISTANCE_FUNCTIONS)
        self.assertIn("euclidean", DISTANCE_FUNCTIONS)
        self.assertIn("manhattan", DISTANCE_FUNCTIONS)
        
        # Verify that the dictionary contains the right functions
        self.assertEqual(DISTANCE_FUNCTIONS["hausdorff"], hausdorff_distance)
        self.assertEqual(DISTANCE_FUNCTIONS["euclidean"], euclidean_distance)
        self.assertEqual(DISTANCE_FUNCTIONS["manhattan"], manhattan_distance)
        
        # Test that the functions in the dictionary work correctly
        self.assertEqual(
            DISTANCE_FUNCTIONS["hausdorff"](self.interval1_1d, self.interval2_1d),
            hausdorff_distance(self.interval1_1d, self.interval2_1d)
        )
        
        self.assertEqual(
            DISTANCE_FUNCTIONS["euclidean"](self.interval1_2d, self.interval2_2d),
            euclidean_distance(self.interval1_2d, self.interval2_2d)
        )
        
        self.assertEqual(
            DISTANCE_FUNCTIONS["manhattan"](self.interval1_3d, self.interval2_3d),
            manhattan_distance(self.interval1_3d, self.interval2_3d)
        )

    def test_edge_cases(self):
        """Test edge cases for distance functions"""
        # Zero-width intervals
        zero_width_1d = np.array([3.0, 3.0])
        zero_width_2d = np.array([[3.0, 3.0], [4.0, 4.0]])
        
        # The distance should still be calculated correctly
        self.assertAlmostEqual(
            euclidean_distance(self.interval1_1d, zero_width_1d),
            np.sqrt((3-1)**2 + (3-5)**2)
        )
        
        # Negative intervals
        negative_interval_1d = np.array([-2.0, -1.0])
        
        self.assertEqual(
            manhattan_distance(negative_interval_1d, self.interval1_1d),
            manhattan_distance(self.interval1_1d, negative_interval_1d)
        )
        
        # Reversed intervals (where min > max)
        # Note: This might not be a valid interval, but testing how the functions handle it
        reversed_interval_1d = np.array([5.0, 1.0])  # Swapped min and max
        
        # The distance functions should work with the values as given, not necessarily
        # treating them as proper intervals
        self.assertEqual(
            hausdorff_distance(reversed_interval_1d, self.interval1_1d),
            max(abs(5-1), abs(1-5))
        )

if __name__ == '__main__':
    unittest.main()
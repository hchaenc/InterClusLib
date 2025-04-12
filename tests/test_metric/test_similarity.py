import unittest
import numpy as np
from interClusLib.metric.similarity import (
    jaccard_similarity,
    dice_similarity,
    bidirectional_similarity_min,
    bidirectional_similarity_prod,
    marginal_similarity,
    SIMILARITY_FUNCTIONS
)

class TestSimilarityFunctions(unittest.TestCase):
    """Test suite for interval similarity functions"""
    
    def setUp(self):
        """Set up test data"""
        # Single-dimensional intervals
        self.identical_1d = np.array([1.0, 5.0])
        self.overlapping_1d_1 = np.array([1.0, 5.0])
        self.overlapping_1d_2 = np.array([3.0, 7.0])
        self.non_overlapping_1d_1 = np.array([1.0, 3.0])
        self.non_overlapping_1d_2 = np.array([4.0, 6.0])
        self.containing_1d_1 = np.array([1.0, 10.0])  # Contains other intervals
        self.containing_1d_2 = np.array([3.0, 5.0])   # Contained by other intervals
        
        # Multi-dimensional intervals (2D)
        self.identical_2d = np.array([[1.0, 5.0], [2.0, 6.0]])
        self.overlapping_2d_1 = np.array([[1.0, 5.0], [2.0, 6.0]])
        self.overlapping_2d_2 = np.array([[3.0, 7.0], [4.0, 8.0]])
        self.non_overlapping_2d_1 = np.array([[1.0, 3.0], [2.0, 4.0]])
        self.non_overlapping_2d_2 = np.array([[4.0, 6.0], [5.0, 7.0]])
        self.mixed_2d_1 = np.array([[1.0, 5.0], [1.0, 3.0]])  # First dim overlaps, second doesn't
        self.mixed_2d_2 = np.array([[3.0, 7.0], [4.0, 6.0]])
        
        # Multi-dimensional intervals (3D)
        self.identical_3d = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
        self.overlapping_3d_1 = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
        self.overlapping_3d_2 = np.array([[3.0, 7.0], [4.0, 8.0], [5.0, 9.0]])

    def test_jaccard_similarity_1d(self):
        """Test Jaccard similarity for 1D intervals"""
        # Identical intervals should have similarity 1.0
        self.assertEqual(jaccard_similarity(self.identical_1d, self.identical_1d), 1.0)
        
        # Overlapping intervals
        # Intersection = [3, 5] = 2, Union = [1, 7] = 6
        # Jaccard = 2/6 = 0.333...
        self.assertAlmostEqual(
            jaccard_similarity(self.overlapping_1d_1, self.overlapping_1d_2),
            2.0/6.0,
            places=5
        )
        
        # Non-overlapping intervals should have similarity 0.0
        self.assertEqual(
            jaccard_similarity(self.non_overlapping_1d_1, self.non_overlapping_1d_2),
            0.0
        )
        
        # Containing intervals
        # Intersection = [3, 5] = 2, Union = [1, 10] = 9
        # Jaccard = 2/9 = 0.222...
        self.assertAlmostEqual(
            jaccard_similarity(self.containing_1d_1, self.containing_1d_2),
            2.0/9.0,
            places=5
        )
        
        # Similarity should be symmetric
        self.assertEqual(
            jaccard_similarity(self.overlapping_1d_1, self.overlapping_1d_2),
            jaccard_similarity(self.overlapping_1d_2, self.overlapping_1d_1)
        )

    def test_jaccard_similarity_2d(self):
        """Test Jaccard similarity for 2D intervals"""
        # Identical intervals should have similarity 1.0
        self.assertEqual(jaccard_similarity(self.identical_2d, self.identical_2d), 1.0)
        
        # Overlapping intervals - average of Jaccard for each dimension
        # Dim 1: Intersection = [3, 5] = 2, Union = [1, 7] = 6, Jaccard = 2/6 = 0.333...
        # Dim 2: Intersection = [4, 6] = 2, Union = [2, 8] = 6, Jaccard = 2/6 = 0.333...
        # Mean = 0.333...
        self.assertAlmostEqual(
            jaccard_similarity(self.overlapping_2d_1, self.overlapping_2d_2),
            (2.0/6.0 + 2.0/6.0)/2.0,
            places=5
        )
        
        # Non-overlapping intervals should have similarity 0.0
        self.assertEqual(
            jaccard_similarity(self.non_overlapping_2d_1, self.non_overlapping_2d_2),
            0.0
        )
        
        # Mixed overlapping/non-overlapping
        # Dim 1: Intersection = [3, 5] = 2, Union = [1, 7] = 6, Jaccard = 2/6 = 0.333...
        # Dim 2: Intersection = 0, Union = 4, Jaccard = 0
        # Mean = (0.333... + 0)/2 = 0.166...
        self.assertAlmostEqual(
            jaccard_similarity(self.mixed_2d_1, self.mixed_2d_2),
            (2.0/6.0 + 0.0)/2.0,
            places=5
        )
        
        # Similarity should be symmetric
        self.assertEqual(
            jaccard_similarity(self.overlapping_2d_1, self.overlapping_2d_2),
            jaccard_similarity(self.overlapping_2d_2, self.overlapping_2d_1)
        )

    def test_dice_similarity_1d(self):
        """Test Dice similarity for 1D intervals"""
        # Identical intervals should have similarity 1.0
        self.assertEqual(dice_similarity(self.identical_1d, self.identical_1d), 1.0)
        
        # Overlapping intervals
        # Intersection = [3, 5] = 2, Sum = 4 + 4 = 8
        # Dice = 2*2/8 = 0.5
        self.assertEqual(
            dice_similarity(self.overlapping_1d_1, self.overlapping_1d_2),
            0.5
        )
        
        # Non-overlapping intervals should have similarity 0.0
        self.assertEqual(
            dice_similarity(self.non_overlapping_1d_1, self.non_overlapping_1d_2),
            0.0
        )
        
        # Containing intervals
        # Intersection = [3, 5] = 2, Sum = 9 + 2 = 11
        # Dice = 2*2/11 = 4/11 ≈ 0.364
        self.assertAlmostEqual(
            dice_similarity(self.containing_1d_1, self.containing_1d_2),
            4.0/11.0,
            places=5
        )
        
        # Similarity should be symmetric
        self.assertEqual(
            dice_similarity(self.overlapping_1d_1, self.overlapping_1d_2),
            dice_similarity(self.overlapping_1d_2, self.overlapping_1d_1)
        )

    def test_bidirectional_similarity_min_1d(self):
        """Test bidirectional min similarity for 1D intervals"""
        # Identical intervals should have similarity 1.0
        self.assertEqual(bidirectional_similarity_min(self.identical_1d, self.identical_1d), 1.0)
        
        # Overlapping intervals
        # Interval 1: [1, 5], Interval 2: [3, 7]
        # Intersection = [3, 5] = 2
        # Interval 1 size = 4, Interval 2 size = 4
        # Overlap ratio 1: 2/4 = 0.5, Overlap ratio 2: 2/4 = 0.5
        # Min = 0.5
        self.assertEqual(
            bidirectional_similarity_min(self.overlapping_1d_1, self.overlapping_1d_2),
            0.5
        )
        
        # Non-overlapping intervals should have similarity 0.0
        self.assertEqual(
            bidirectional_similarity_min(self.non_overlapping_1d_1, self.non_overlapping_1d_2),
            0.0
        )
        
        # Containing intervals
        # Interval 1: [1, 10], Interval 2: [3, 5]
        # Intersection = [3, 5] = 2
        # Interval 1 size = 9, Interval 2 size = 2
        # Overlap ratio 1: 2/9 ≈ 0.222, Overlap ratio 2: 2/2 = 1.0
        # Min = 0.222...
        self.assertAlmostEqual(
            bidirectional_similarity_min(self.containing_1d_1, self.containing_1d_2),
            2.0/9.0,
            places=5
        )
        
        # Similarity should be symmetric
        self.assertEqual(
            bidirectional_similarity_min(self.overlapping_1d_1, self.overlapping_1d_2),
            bidirectional_similarity_min(self.overlapping_1d_2, self.overlapping_1d_1)
        )

    def test_bidirectional_similarity_prod_1d(self):
        """Test bidirectional prod similarity for 1D intervals"""
        # Identical intervals should have similarity 1.0
        self.assertEqual(bidirectional_similarity_prod(self.identical_1d, self.identical_1d), 1.0)
        
        # Overlapping intervals
        # Interval 1: [1, 5], Interval 2: [3, 7]
        # Intersection = [3, 5] = 2
        # Interval 1 size = 4, Interval 2 size = 4
        # Overlap ratio 1: 2/4 = 0.5, Overlap ratio 2: 2/4 = 0.5
        # Product = 0.5 * 0.5 = 0.25
        self.assertEqual(
            bidirectional_similarity_prod(self.overlapping_1d_1, self.overlapping_1d_2),
            0.25
        )
        
        # Non-overlapping intervals should have similarity 0.0
        self.assertEqual(
            bidirectional_similarity_prod(self.non_overlapping_1d_1, self.non_overlapping_1d_2),
            0.0
        )
        
        # Containing intervals
        # Interval 1: [1, 10], Interval 2: [3, 5]
        # Intersection = [3, 5] = 2
        # Interval 1 size = 9, Interval 2 size = 2
        # Overlap ratio 1: 2/9 ≈ 0.222, Overlap ratio 2: 2/2 = 1.0
        # Product = 0.222... * 1.0 = 0.222...
        self.assertAlmostEqual(
            bidirectional_similarity_prod(self.containing_1d_1, self.containing_1d_2),
            2.0/9.0,
            places=5
        )
        
        # Similarity should be symmetric
        self.assertEqual(
            bidirectional_similarity_prod(self.overlapping_1d_1, self.overlapping_1d_2),
            bidirectional_similarity_prod(self.overlapping_1d_2, self.overlapping_1d_1)
        )

    def test_marginal_similarity_1d(self):
        """Test marginal similarity for 1D intervals"""
        # Identical intervals should have similarity 1.0
        self.assertEqual(marginal_similarity(self.identical_1d, self.identical_1d), 1.0)
        
        # Overlapping intervals
        # Interval 1: [1, 5], Interval 2: [3, 7]
        # Intersection = [3, 5] = 2, Union = [1, 7] = 6, distance = 0, domain = 6
        # Marginal = 0.5 * (2/6 + 1 - 0/6) = 0.5 * (0.333... + 1) ≈ 0.667
        self.assertAlmostEqual(
            marginal_similarity(self.overlapping_1d_1, self.overlapping_1d_2),
            0.5 * (2.0/6.0 + 1.0),
            places=5
        )
        
        # Non-overlapping intervals
        # Interval 1: [1, 3], Interval 2: [4, 6]
        # Intersection = 0, Union = 4, distance = 1, domain = 5
        # Marginal = 0.5 * (0/4 + 1 - 1/5) = 0.5 * (0 + 0.8) = 0.4
        self.assertEqual(
            marginal_similarity(self.non_overlapping_1d_1, self.non_overlapping_1d_2),
            0.5 * (0.0 + 1.0 - 1.0/5.0)
        )
        
        # Similarity should be symmetric
        self.assertEqual(
            marginal_similarity(self.overlapping_1d_1, self.overlapping_1d_2),
            marginal_similarity(self.overlapping_1d_2, self.overlapping_1d_1)
        )

    def test_similarity_functions_dictionary(self):
        """Test the SIMILARITY_FUNCTIONS dictionary"""
        # Check that all defined functions are in the dictionary
        self.assertIn("jaccard", SIMILARITY_FUNCTIONS)
        self.assertIn("dice", SIMILARITY_FUNCTIONS)
        self.assertIn("bidirectional_min", SIMILARITY_FUNCTIONS)
        self.assertIn("bidirectional_prod", SIMILARITY_FUNCTIONS)
        self.assertIn("marginal", SIMILARITY_FUNCTIONS)
        
        # Verify that the dictionary contains the right functions
        self.assertEqual(SIMILARITY_FUNCTIONS["jaccard"], jaccard_similarity)
        self.assertEqual(SIMILARITY_FUNCTIONS["dice"], dice_similarity)
        self.assertEqual(SIMILARITY_FUNCTIONS["bidirectional_min"], bidirectional_similarity_min)
        self.assertEqual(SIMILARITY_FUNCTIONS["bidirectional_prod"], bidirectional_similarity_prod)
        self.assertEqual(SIMILARITY_FUNCTIONS["marginal"], marginal_similarity)
        
        # Test that the functions in the dictionary work correctly
        self.assertEqual(
            SIMILARITY_FUNCTIONS["jaccard"](self.identical_1d, self.identical_1d),
            jaccard_similarity(self.identical_1d, self.identical_1d)
        )

    def test_edge_cases(self):
        """Test edge cases for similarity functions"""
        # Zero-width intervals
        zero_width_1d = np.array([3.0, 3.0])
        
        # When one interval has zero width, Jaccard similarity should handle it gracefully
        # Expected to be 0 since intersection is 0 and union is the other interval's width
        self.assertEqual(
            jaccard_similarity(zero_width_1d, self.overlapping_1d_1),
            0.0
        )
        
        # Negative intervals (where max < min) - testing edge handling
        negative_interval_1d = np.array([5.0, 1.0])  # Reversed bounds
        
        # For simplicity, we just check if functions run without error
        # The results may not be semantically meaningful with negative intervals
        _ = jaccard_similarity(negative_interval_1d, self.overlapping_1d_1)
        _ = dice_similarity(negative_interval_1d, self.overlapping_1d_1)
        _ = bidirectional_similarity_min(negative_interval_1d, self.overlapping_1d_1)
        _ = bidirectional_similarity_prod(negative_interval_1d, self.overlapping_1d_1)
        _ = marginal_similarity(negative_interval_1d, self.overlapping_1d_1)
        
        # Test with 3D intervals
        self.assertAlmostEqual(
            jaccard_similarity(self.overlapping_3d_1, self.overlapping_3d_2),
            (2.0/6.0 + 2.0/6.0 + 2.0/6.0)/3.0,  # Average of Jaccard in each dimension
            places=5
        )

    def test_boundary_values(self):
        """Test similarity functions with boundary values"""
        # Create intervals that have boundary cases
        # Case 1: Intervals that touch but don't overlap
        touching_1d_1 = np.array([1.0, 3.0])
        touching_1d_2 = np.array([3.0, 5.0])
        
        # For Jaccard, touching intervals have no intersection
        self.assertEqual(jaccard_similarity(touching_1d_1, touching_1d_2), 0.0)
        
        # Case 2: One interval inside another
        inner_1d = np.array([2.0, 3.0])
        outer_1d = np.array([1.0, 4.0])
        
        # For bidirectional, the smaller interval is fully contained
        self.assertEqual(
            bidirectional_similarity_min(inner_1d, outer_1d),
            min(1.0, 1.0/3.0)  # Min of (1.0, intersection/outer_size)
        )
        
        # Case 3: Empty interval (invalid, but testing robustness)
        empty_interval = np.array([0.0, 0.0])
        
        # Similarity with empty interval should not cause division by zero
        self.assertEqual(jaccard_similarity(empty_interval, self.overlapping_1d_1), 0.0)
        self.assertEqual(dice_similarity(empty_interval, self.overlapping_1d_1), 0.0)

if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np
from interClusLib.preprocessing import fix_inverted_intervals

class TestFixInvertedIntervals(unittest.TestCase):
    """Test suite for the fix_inverted_intervals function."""

    def setUp(self):
        """Set up test data."""
        # Create a sample dataset with some inverted intervals
        self.interval_data = np.array([
            # Sample 1: No inversions
            [[1.0, 3.0], [2.0, 5.0], [0.0, 4.0]],
            
            # Sample 2: First dimension inverted
            [[5.0, 2.0], [1.0, 3.0], [0.0, 2.0]],
            
            # Sample 3: Last dimension inverted
            [[1.0, 4.0], [2.0, 6.0], [7.0, 3.0]],
            
            # Sample 4: All dimensions inverted
            [[8.0, 3.0], [9.0, 4.0], [6.0, 2.0]]
        ])
        
        # Expected inverted mask
        self.expected_mask = np.array([
            [False, False, False],  # Sample 1
            [True, False, False],   # Sample 2
            [False, False, True],   # Sample 3
            [True, True, True]      # Sample 4
        ])
        
        # Create data with None values
        self.data_with_none = np.array([
            # Sample 1: Normal interval
            [[1.0, 3.0], [2.0, 5.0]],
            
            # Sample 2: Inverted interval
            [[5.0, 2.0], [1.0, 3.0]],
            
            # Sample 3: Lower bound is None
            [[None, 4.0], [2.0, 6.0]],
            
            # Sample 4: Upper bound is None
            [[1.0, None], [3.0, 7.0]]
        ], dtype=object)

    def test_auto_detect_inverted(self):
        """Test automatic detection of inverted intervals."""
        # Fix intervals using 'swap' method
        fixed_data, inverted_mask = fix_inverted_intervals(self.interval_data, method='swap')
        
        # Check that the inverted mask was correctly identified
        np.testing.assert_array_equal(inverted_mask, self.expected_mask)
        
        # For all previously inverted intervals, check that they are now fixed (lower < upper)
        for i in range(len(self.interval_data)):
            for j in range(self.interval_data.shape[1]):
                if inverted_mask[i, j]:
                    self.assertLess(fixed_data[i, j, 0], fixed_data[i, j, 1])

    def test_swap_method(self):
        """Test the 'swap' method."""
        # Fix intervals using 'swap' method
        fixed_data, _ = fix_inverted_intervals(self.interval_data, method='swap')
        
        # Check that formerly inverted intervals are now correct
        # Sample 2, dimension 0: [5.0, 2.0] should become [2.0, 5.0]
        self.assertEqual(fixed_data[1, 0, 0], 2.0)
        self.assertEqual(fixed_data[1, 0, 1], 5.0)
        
        # Sample 3, dimension 2: [7.0, 3.0] should become [3.0, 7.0]
        self.assertEqual(fixed_data[2, 2, 0], 3.0)
        self.assertEqual(fixed_data[2, 2, 1], 7.0)
        
        # Sample 4, all dimensions should be swapped
        self.assertEqual(fixed_data[3, 0, 0], 3.0)
        self.assertEqual(fixed_data[3, 0, 1], 8.0)
        self.assertEqual(fixed_data[3, 1, 0], 4.0)
        self.assertEqual(fixed_data[3, 1, 1], 9.0)
        self.assertEqual(fixed_data[3, 2, 0], 2.0)
        self.assertEqual(fixed_data[3, 2, 1], 6.0)
        
        # Non-inverted intervals should remain unchanged
        # Sample 1, all dimensions
        np.testing.assert_array_equal(fixed_data[0], self.interval_data[0])
        # Sample 2, dimensions 1 and 2
        np.testing.assert_array_equal(fixed_data[1, 1:], self.interval_data[1, 1:])
        # Sample 3, dimensions 0 and 1
        np.testing.assert_array_equal(fixed_data[2, :2], self.interval_data[2, :2])

    def test_remove_method(self):
        """Test the 'remove' method."""
        # Fix intervals using 'remove' method
        fixed_data, _ = fix_inverted_intervals(self.interval_data, method='remove')
        
        # Only sample 1 (no inversions) should remain
        self.assertEqual(fixed_data.shape[0], 1)
        np.testing.assert_array_equal(fixed_data[0], self.interval_data[0])

    def test_with_provided_mask(self):
        """Test with a provided inverted mask."""
        # Create a custom mask where only the first dimension of sample 4 is considered inverted
        custom_mask = np.zeros_like(self.expected_mask, dtype=bool)
        custom_mask[3, 0] = True
        
        # Fix intervals using the custom mask
        fixed_data, returned_mask = fix_inverted_intervals(
            self.interval_data, method='swap', inverted_mask=custom_mask)
        
        # Check that the returned mask is the same as the provided one
        np.testing.assert_array_equal(returned_mask, custom_mask)
        
        # Only sample 4, dimension 0 should be swapped
        self.assertEqual(fixed_data[3, 0, 0], 3.0)
        self.assertEqual(fixed_data[3, 0, 1], 8.0)
        
        # Other dimensions of sample 4 should remain inverted
        self.assertEqual(fixed_data[3, 1, 0], 9.0)
        self.assertEqual(fixed_data[3, 1, 1], 4.0)
        self.assertEqual(fixed_data[3, 2, 0], 6.0)
        self.assertEqual(fixed_data[3, 2, 1], 2.0)

    def test_with_none_values(self):
        """Test handling of None values."""
        # Fix intervals with None values
        fixed_data, inverted_mask = fix_inverted_intervals(self.data_with_none, method='swap')
        
        # Check inverted intervals detection
        # Only sample 2, dimension 0 should be detected as inverted
        expected_none_mask = np.array([
            [False, False],  # Sample 1
            [True, False],   # Sample 2
            [False, False],  # Sample 3 (None in lower bound)
            [False, False]   # Sample 4 (None in upper bound)
        ])
        np.testing.assert_array_equal(inverted_mask, expected_none_mask)
        
        # Check that the inverted interval was fixed
        self.assertEqual(fixed_data[1, 0, 0], 2.0)
        self.assertEqual(fixed_data[1, 0, 1], 5.0)
        
        # Check that intervals with None values remain unchanged
        self.assertIs(fixed_data[2, 0, 0], None)
        self.assertEqual(fixed_data[2, 0, 1], 4.0)
        self.assertEqual(fixed_data[3, 0, 0], 1.0)
        self.assertIs(fixed_data[3, 0, 1], None)

    def test_no_inverted_intervals(self):
        """Test case with no inverted intervals."""
        # Create data with no inversions
        no_inversions_data = np.array([
            [[1.0, 3.0], [2.0, 5.0]],
            [[2.0, 4.0], [3.0, 6.0]]
        ])
        
        # Fix intervals, but there's nothing to fix
        fixed_data, inverted_mask = fix_inverted_intervals(no_inversions_data, method='swap')
        
        # Check that no intervals were identified as inverted
        np.testing.assert_array_equal(inverted_mask, np.zeros((2, 2), dtype=bool))
        
        # Check that data remains unchanged
        np.testing.assert_array_equal(fixed_data, no_inversions_data)

    def test_edge_cases(self):
        """Test edge cases."""
        # Test with equal bounds (not considered inverted)
        equal_bounds_data = np.array([
            [[1.0, 1.0], [2.0, 2.0]],
            [[3.0, 3.0], [4.0, 4.0]]
        ])
        
        fixed_data, inverted_mask = fix_inverted_intervals(equal_bounds_data, method='swap')
        
        # Check that no intervals were identified as inverted
        np.testing.assert_array_equal(inverted_mask, np.zeros((2, 2), dtype=bool))
        
        # Check that data remains unchanged
        np.testing.assert_array_equal(fixed_data, equal_bounds_data)

if __name__ == '__main__':
    unittest.main()
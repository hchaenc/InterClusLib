import unittest
import numpy as np
from interClusLib.preprocessing import (
    min_max_normalize,
    midpoint_width_normalize,
    robust_quantile_normalize,
    z_score_normalize
)

class TestNormalizationFunctions(unittest.TestCase):
    """Test suite for interval data normalization functions."""

    def setUp(self):
        """Set up test data."""
        # Create a sample interval dataset with 5 samples and 2 dimensions
        self.interval_data = np.array([
            [[1.0, 3.0], [10.0, 15.0]],  # Sample 1, 2 dimensions
            [[2.0, 6.0], [12.0, 14.0]],  # Sample 2
            [[5.0, 8.0], [8.0, 16.0]],   # Sample 3
            [[4.0, 7.0], [5.0, 10.0]],   # Sample 4
            [[0.0, 2.0], [20.0, 25.0]]   # Sample 5
        ])
        
        # Data with extreme outliers for testing robust methods
        self.data_with_outliers = np.array([
            [[1.0, 3.0], [10.0, 15.0]],  # Regular data
            [[2.0, 6.0], [12.0, 14.0]],
            [[5.0, 8.0], [8.0, 16.0]],
            [[4.0, 7.0], [5.0, 10.0]],
            [[0.0, 2.0], [20.0, 25.0]],
            [[100.0, 150.0], [200.0, 300.0]]  # Outlier
        ])

    def test_min_max_normalize(self):
        """Test min-max normalization."""
        # Normalize to default range [0, 1]
        normalized_data = min_max_normalize(self.interval_data)
        
        # Check output shape
        self.assertEqual(normalized_data.shape, self.interval_data.shape)
        
        # Check that values are in range [0, 1]
        self.assertTrue(np.all(normalized_data >= 0))
        self.assertTrue(np.all(normalized_data <= 1))
        
        # Check that min-max scaling was applied correctly for each dimension
        for d in range(self.interval_data.shape[1]):
            # Original min/max values
            orig_min = np.min(self.interval_data[:, d, 0])
            orig_max = np.max(self.interval_data[:, d, 1])
            
            # Check that the original min maps to 0 and max maps to 1
            # Find where the min occurs in the original data
            min_idx = np.where(self.interval_data[:, d, 0] == orig_min)[0][0]
            max_idx = np.where(self.interval_data[:, d, 1] == orig_max)[0][0]
            
            self.assertAlmostEqual(normalized_data[min_idx, d, 0], 0.0)
            self.assertAlmostEqual(normalized_data[max_idx, d, 1], 1.0)
        
        # Test with custom range [-1, 1]
        custom_range = (-1, 1)
        normalized_data_custom = min_max_normalize(self.interval_data, feature_range=custom_range)
        
        # Check that values are in range [-1, 1]
        self.assertTrue(np.all(normalized_data_custom >= -1))
        self.assertTrue(np.all(normalized_data_custom <= 1))
        
        # Check that relative relationships are preserved
        # If interval A is wider than interval B in the original data,
        # the same should be true in the normalized data
        for d in range(self.interval_data.shape[1]):
            orig_widths = self.interval_data[:, d, 1] - self.interval_data[:, d, 0]
            norm_widths = normalized_data[:, d, 1] - normalized_data[:, d, 0]
            
            # Sort samples by original width
            sorted_by_orig = np.argsort(orig_widths)
            # Check if normalized widths maintain the same order
            sorted_by_norm = np.argsort(norm_widths)
            np.testing.assert_array_equal(sorted_by_orig, sorted_by_norm)

    def test_midpoint_width_normalize(self):
        """Test midpoint-width normalization."""
        normalized_data = midpoint_width_normalize(self.interval_data)
        
        # Check output shape
        self.assertEqual(normalized_data.shape, self.interval_data.shape)
        
        # Verify that midpoints and widths are normalized independently
        for d in range(self.interval_data.shape[1]):
            # Original midpoints and widths
            orig_midpoints = (self.interval_data[:, d, 0] + self.interval_data[:, d, 1]) / 2
            orig_widths = self.interval_data[:, d, 1] - self.interval_data[:, d, 0]
            
            # Normalized midpoints and widths
            norm_midpoints = (normalized_data[:, d, 0] + normalized_data[:, d, 1]) / 2
            norm_widths = normalized_data[:, d, 1] - normalized_data[:, d, 0]
            
            # Check that the midpoint with min value maps to 0 and max to 1
            min_mid_idx = np.argmin(orig_midpoints)
            max_mid_idx = np.argmax(orig_midpoints)
            self.assertAlmostEqual(norm_midpoints[min_mid_idx], 0.0)
            self.assertAlmostEqual(norm_midpoints[max_mid_idx], 1.0)
            
            # Check that the width with min value maps to 0 and max to 1
            min_width_idx = np.argmin(orig_widths)
            max_width_idx = np.argmax(orig_widths)
            self.assertAlmostEqual(norm_widths[min_width_idx], 0.0)
            self.assertAlmostEqual(norm_widths[max_width_idx], 1.0)

    def test_robust_quantile_normalize(self):
        """Test robust quantile normalization."""
        # Test with default quantiles (0.05, 0.95)
        normalized_data = robust_quantile_normalize(self.data_with_outliers)
        
        # Check output shape
        self.assertEqual(normalized_data.shape, self.data_with_outliers.shape)
        
        # Check that values are clipped to [0, 1]
        self.assertTrue(np.all(normalized_data >= 0))
        self.assertTrue(np.all(normalized_data <= 1))
        
        # Test with custom quantiles (0.1, 0.9)
        normalized_data_custom = robust_quantile_normalize(
            self.data_with_outliers, q_low=0.1, q_high=0.9)
        
        # Check that more values are clipped with tighter quantiles
        n_clipped_default = np.sum((normalized_data == 0) | (normalized_data == 1))
        n_clipped_custom = np.sum((normalized_data_custom == 0) | (normalized_data_custom == 1))
        self.assertGreaterEqual(n_clipped_custom, n_clipped_default)

    def test_z_score_normalize(self):
        """Test z-score normalization."""
        normalized_data = z_score_normalize(self.interval_data)
        
        # Check output shape
        self.assertEqual(normalized_data.shape, self.interval_data.shape)
        
        # Verify that midpoints follow z-score properties
        for d in range(self.interval_data.shape[1]):
            # Original midpoints
            orig_midpoints = (self.interval_data[:, d, 0] + self.interval_data[:, d, 1]) / 2
            
            # Normalized midpoints
            norm_midpoints = (normalized_data[:, d, 0] + normalized_data[:, d, 1]) / 2
            
            # Check that mean is approximately 0 and std is approximately 1
            self.assertAlmostEqual(np.mean(norm_midpoints), 0, places=10)
            self.assertAlmostEqual(np.std(norm_midpoints), 1, places=10)
            
            # Check that width scale transformation is consistent
            # We don't check exact ordering preservation, as z-score transformation
            # can legitimately change relative widths
            
            # Instead, verify that width scale is related to the std of original midpoints
            orig_widths = self.interval_data[:, d, 1] - self.interval_data[:, d, 0]
            norm_widths = normalized_data[:, d, 1] - normalized_data[:, d, 0]
            
            # The ratio of normalized width to original width should be roughly constant
            # (though not exactly, due to how z-score transforms data)
            width_ratios = norm_widths / orig_widths
            
            # Check that the width ratios don't vary too much (allowing some flexibility)
            self.assertLess(np.std(width_ratios), 0.1, 
                            msg=f"Width ratios should be relatively consistent: {width_ratios}")
            
            # Also check that very large original intervals remain larger than very small ones
            # after normalization (checking extreme cases only)
            if len(orig_widths) > 2:
                max_orig_idx = np.argmax(orig_widths)
                min_orig_idx = np.argmin(orig_widths)
                self.assertGreater(norm_widths[max_orig_idx], norm_widths[min_orig_idx],
                                msg="The largest interval should still be larger than the smallest after normalization")

    def test_edge_cases(self):
        """Test edge cases for normalization functions."""
        # Test with constant values (should handle division by zero gracefully)
        constant_data = np.array([
            [[5.0, 10.0], [20.0, 25.0]],
            [[5.0, 10.0], [20.0, 25.0]],
            [[5.0, 10.0], [20.0, 25.0]]
        ])
        
        # For min-max, if all values are the same, they should map to 0
        # But we handle this by checking that the function runs without errors
        try:
            min_max_result = min_max_normalize(constant_data)
            # All values in each dimension should be equal
            self.assertTrue(np.allclose(min_max_result[:, 0, 0], min_max_result[0, 0, 0]))
            self.assertTrue(np.allclose(min_max_result[:, 0, 1], min_max_result[0, 0, 1]))
        except Exception as e:
            self.fail(f"min_max_normalize raised {type(e).__name__} unexpectedly!")
        
        # Similarly for other functions
        try:
            midpoint_width_result = midpoint_width_normalize(constant_data)
            z_score_result = z_score_normalize(constant_data)
            robust_result = robust_quantile_normalize(constant_data)
        except Exception as e:
            self.fail(f"Normalization function raised {type(e).__name__} unexpectedly!")
        
        # Test with single sample
        single_sample = np.array([[[1.0, 3.0], [5.0, 10.0]]])
        
        try:
            min_max_single = min_max_normalize(single_sample)
            midpoint_width_single = midpoint_width_normalize(single_sample)
            z_score_single = z_score_normalize(single_sample)
            robust_single = robust_quantile_normalize(single_sample)
        except Exception as e:
            self.fail(f"Normalization with single sample raised {type(e).__name__} unexpectedly!")

if __name__ == '__main__':
    unittest.main()
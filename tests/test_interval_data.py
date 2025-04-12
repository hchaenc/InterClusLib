import unittest
import pandas as pd
import numpy as np
from interClusLib.IntervalData import IntervalData

class TestIntervalData(unittest.TestCase):
    """Test suite for the IntervalData class."""

    def setUp(self):
        """Set up test data for each test method."""
        # Create a sample DataFrame with interval data
        self.sample_data = pd.DataFrame({
            'feature_1_lower': [1.0, 2.0, 3.0],
            'feature_1_upper': [4.0, 5.0, 6.0],
            'feature_2_lower': [0.5, 1.5, 2.5],
            'feature_2_upper': [3.5, 4.5, 5.5]
        })
        
        # DataFrame with missing values
        self.data_with_missing = pd.DataFrame({
            'feature_1_lower': [1.0, np.nan, 3.0],
            'feature_1_upper': [4.0, 5.0, np.nan],
            'feature_2_lower': [0.5, 1.5, 2.5],
            'feature_2_upper': [3.5, np.nan, 5.5]
        })
        
        # DataFrame with invalid intervals (lower > upper)
        self.data_with_invalid = pd.DataFrame({
            'feature_1_lower': [1.0, 6.0, 3.0],
            'feature_1_upper': [4.0, 5.0, 6.0],
            'feature_2_lower': [0.5, 1.5, 2.5],
            'feature_2_upper': [3.5, 4.5, 5.5]
        })

    def test_initialization(self):
        """Test initialization of IntervalData."""
        # Test with valid data
        interval_data = IntervalData(self.sample_data)
        self.assertIsInstance(interval_data, IntervalData)
        self.assertEqual(len(interval_data.features), 2)
        self.assertEqual(len(interval_data.columns), 4)
        self.assertFalse(interval_data.has_missing_values)
        
        # Test with non-DataFrame input
        with self.assertRaises(ValueError):
            IntervalData([1, 2, 3])

    def test_handle_missing_values(self):
        """Test handling of missing values."""
        # Test with missing values and handle_missing=True
        interval_data = IntervalData(self.data_with_missing, handle_missing=True)
        self.assertTrue(interval_data.has_missing_values)
        
        # Test with missing values and handle_missing=False
        with self.assertRaises(ValueError):
            IntervalData(self.data_with_missing, handle_missing=False)

    def test_get_intervals(self):
        """Test the get_intervals method."""
        # Test with valid data
        interval_data = IntervalData(self.sample_data)
        intervals = interval_data.get_intervals()
        
        # Check shape: [n_samples, n_features, 2] where 2 represents [lower, upper]
        self.assertEqual(intervals.shape, (3, 2, 2))
        
        # Check specific values
        self.assertEqual(intervals[0, 0, 0], 1.0)  # First sample, first feature, lower bound
        self.assertEqual(intervals[0, 0, 1], 4.0)  # First sample, first feature, upper bound
        self.assertEqual(intervals[1, 1, 0], 1.5)  # Second sample, second feature, lower bound
        
        # Test with missing values
        interval_data_missing = IntervalData(self.data_with_missing)
        intervals_missing = interval_data_missing.get_intervals()
        
        # Check that shape is correct
        self.assertEqual(intervals_missing.shape, (3, 2, 2))
        
        # Check that NaN values are preserved
        self.assertTrue(np.isnan(intervals_missing[1, 0, 0]))  # Second sample, first feature, lower bound
        self.assertTrue(np.isnan(intervals_missing[2, 0, 1]))  # Third sample, first feature, upper bound

    def test_validate_intervals(self):
        """Test the validate_intervals method."""
        # Test with valid intervals
        interval_data = IntervalData(self.sample_data)
        validated = interval_data.validate_intervals()
        self.assertIs(validated, interval_data)  # Should return self if no fixing needed
        
        # Test with invalid intervals and no fixing
        invalid_data = IntervalData(self.data_with_invalid)
        validated = invalid_data.validate_intervals(fix_invalid=False)
        self.assertIs(validated, invalid_data)  # Should return self if not fixing
        
        # Test with invalid intervals and fixing
        fixed = invalid_data.validate_intervals(fix_invalid=True)
        self.assertIsInstance(fixed, IntervalData)  # Should return a new instance
        
        # Check that values were swapped
        fixed_df = fixed.to_dataframe()
        self.assertEqual(fixed_df.loc[1, 'feature_1_lower'], 5.0)
        self.assertEqual(fixed_df.loc[1, 'feature_1_upper'], 6.0)

    def test_features_extraction(self):
        """Test extraction of feature names."""
        interval_data = IntervalData(self.sample_data)
        
        # Check feature names
        self.assertEqual(interval_data.features, ['feature_1', 'feature_2'])
        
        # Test with additional non-interval column
        data_with_extra = self.sample_data.copy()
        data_with_extra['other_column'] = [10, 20, 30]
        interval_data_extra = IntervalData(data_with_extra)
        
        # Should still only extract interval features
        self.assertEqual(interval_data_extra.features, ['feature_1', 'feature_2'])
        
        # Check all columns
        self.assertIn('other_column', interval_data_extra.columns)

    def test_random_data_generation(self):
        """Test the random_data class method."""
        # Generate random data
        n_samples = 10
        n_features = 3
        random_interval_data = IntervalData.random_data(n_samples, n_features)
        
        # Check if it's a valid IntervalData object
        self.assertIsInstance(random_interval_data, IntervalData)
        
        # Check dimensions
        intervals = random_interval_data.get_intervals()
        self.assertEqual(intervals.shape, (n_samples, n_features, 2))
        
        # Check that all lower bounds are less than or equal to upper bounds
        for i in range(n_samples):
            for j in range(n_features):
                self.assertLessEqual(intervals[i, j, 0], intervals[i, j, 1])

    def test_make_interval_blobs(self):
        """Test the make_interval_blobs class method."""
        # Generate interval blobs
        n_samples = 20
        n_clusters = 2
        n_dims = 2
        blob_data = IntervalData.make_interval_blobs(
            n_samples=n_samples, 
            n_clusters=n_clusters,
            n_dims=n_dims,
            random_state=42
        )
        
        # Check if it's a valid IntervalData object
        self.assertIsInstance(blob_data, IntervalData)
        
        # Check dimensions
        intervals = blob_data.get_intervals()
        self.assertEqual(intervals.shape, (n_samples, n_dims, 2))
        
        # Test with explicit samples per cluster
        samples_per_cluster = [5, 10]
        blob_data2 = IntervalData.make_interval_blobs(
            n_samples=samples_per_cluster,
            n_dims=2,
            random_state=42
        )
        
        intervals2 = blob_data2.get_intervals()
        self.assertEqual(intervals2.shape, (sum(samples_per_cluster), 2, 2))

    def test_to_dataframe(self):
        """Test the to_dataframe method."""
        interval_data = IntervalData(self.sample_data)
        df = interval_data.to_dataframe()
        
        # Should return the original DataFrame
        pd.testing.assert_frame_equal(df, self.sample_data)

    def test_check_missing_values(self):
        """Test the _check_missing_values method."""
        # Data without missing values
        interval_data = IntervalData(self.sample_data)
        self.assertFalse(interval_data.has_missing_values)
        
        # Data with missing values
        interval_data_missing = IntervalData(self.data_with_missing)
        self.assertTrue(interval_data_missing.has_missing_values)
        
        # Check that it returns a Python native bool
        self.assertIs(type(interval_data.has_missing_values), bool)
        self.assertIs(type(interval_data_missing.has_missing_values), bool)

if __name__ == '__main__':
    unittest.main()
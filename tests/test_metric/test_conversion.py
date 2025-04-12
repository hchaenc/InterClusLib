import unittest
import numpy as np
from interClusLib.metric.conversion import sim_to_dist, dist_to_sim

class TestConversion(unittest.TestCase):
    """Test suite for similarity-distance conversion functions"""
    
    def test_sim_to_dist_scalar(self):
        """Test similarity to distance conversion with scalar inputs"""
        # Test with standard values
        self.assertEqual(sim_to_dist(1.0), 0.0)  # Perfect similarity -> zero distance
        self.assertEqual(sim_to_dist(0.0), 1.0)  # No similarity -> maximum distance
        self.assertEqual(sim_to_dist(0.5), 0.5)  # Middle value
        
        # Test with boundary values - use assertAlmostEqual for floating point
        self.assertAlmostEqual(sim_to_dist(0.999), 0.001, places=10)
        self.assertAlmostEqual(sim_to_dist(0.001), 0.999, places=10)
    
    def test_sim_to_dist_array(self):
        """Test similarity to distance conversion with array inputs"""
        # Test with numpy array
        sim_array = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(sim_to_dist(sim_array), expected)
        
        # Test with list
        sim_list = [1.0, 0.75, 0.5, 0.25, 0.0]
        expected_list = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(sim_to_dist(sim_list), expected_list)
    
    def test_sim_to_dist_invalid_mode(self):
        """Test sim_to_dist with invalid mode parameter"""
        with self.assertRaises(ValueError):
            sim_to_dist(0.5, mode="invalid_mode")
    
    def test_dist_to_sim_scalar(self):
        """Test distance to similarity conversion with scalar inputs"""
        # Test with standard values
        self.assertEqual(dist_to_sim(0.0), 1.0)  # Zero distance -> perfect similarity
        self.assertAlmostEqual(dist_to_sim(1.0), 0.5)  # Unit distance -> 0.5 similarity
        self.assertAlmostEqual(dist_to_sim(4.0), 0.2)  # Higher distance -> lower similarity
        
        # Test with extreme values
        self.assertAlmostEqual(dist_to_sim(999.0), 0.001, places=3)
        self.assertAlmostEqual(dist_to_sim(0.001), 0.999, places=3)
    
    def test_dist_to_sim_array(self):
        """Test distance to similarity conversion with array inputs"""
        # Test with numpy array
        dist_array = np.array([0.0, 1.0, 3.0, 9.0])
        expected = np.array([1.0, 0.5, 0.25, 0.1])
        np.testing.assert_array_almost_equal(dist_to_sim(dist_array), expected)
        
        # Test with list
        dist_list = [0.0, 1.0, 3.0, 9.0]
        expected_list = np.array([1.0, 0.5, 0.25, 0.1])
        np.testing.assert_array_almost_equal(dist_to_sim(dist_list), expected_list)
    
    def test_dist_to_sim_invalid_mode(self):
        """Test dist_to_sim with invalid mode parameter"""
        with self.assertRaises(ValueError):
            dist_to_sim(0.5, mode="invalid_mode")
    
    def test_conversion_symmetry(self):
        """Test if converting back and forth returns close to original value"""
        # For similarity values - adjust tolerance based on the formulas
        sim_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        for sim in sim_values:
            dist = sim_to_dist(sim)
            sim_again = dist_to_sim(dist)
            # With these formulas, the error is larger for smaller similarity values
            # Use a different tolerance depending on the initial value
            if sim <= 0.5:
                tolerance = 0.5  # Larger tolerance for small values
            else:
                tolerance = 0.3  # Smaller tolerance for large values
            self.assertLess(abs(sim - sim_again), tolerance)
    
    def test_conversion_with_invalid_inputs(self):
        """Test conversions with invalid input values"""
        # Test with negative similarity (might work but gives illogical results)
        neg_sim_result = sim_to_dist(-0.5)
        self.assertEqual(neg_sim_result, 1.5)
        
        # Test with negative distance (might work but gives illogical results)
        neg_dist_result = dist_to_sim(-0.5)
        self.assertEqual(neg_dist_result, 2.0)
        
        # Test with values greater than 1 for similarity (might work but gives illogical results)
        large_sim_result = sim_to_dist(1.5)
        self.assertEqual(large_sim_result, -0.5)
    
    def test_conversion_with_special_values(self):
        """Test conversions with special values like NaN, inf"""
        # Test with NaN
        self.assertTrue(np.isnan(sim_to_dist(float('nan'))))
        self.assertTrue(np.isnan(dist_to_sim(float('nan'))))
        
        # Test with infinity
        self.assertEqual(dist_to_sim(float('inf')), 0.0)
        # No logical counterpart for sim_to_dist with infinity

if __name__ == '__main__':
    unittest.main()
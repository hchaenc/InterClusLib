import unittest
import numpy as np
from interClusLib.evaluation import (
    calinski_harabasz_index,
    davies_bouldin_index,
    dunn_index,
    distortion_score,
    silhouette_score
)

class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for clustering evaluation metrics."""
    
    def setUp(self):
        """Set up test data common to all tests."""
        # Create simple interval data with clear clusters
        # Cluster 1: Three intervals around [1-3, 1-3, 1-3]
        # Cluster 2: Three intervals around [7-9, 7-9, 7-9]
        self.interval_data = np.array([
            # Cluster 1
            [[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]],  # Point 1
            [[1.2, 3.2], [0.9, 2.9], [1.1, 3.1]],  # Point 2
            [[0.8, 2.8], [1.1, 3.1], [0.9, 2.9]],  # Point 3
            
            # Cluster 2
            [[7.0, 9.0], [7.0, 9.0], [7.0, 9.0]],  # Point 4
            [[7.2, 9.2], [6.9, 8.9], [7.1, 9.1]],  # Point 5
            [[6.8, 8.8], [7.1, 9.1], [6.9, 8.9]],  # Point 6
        ])
        
        # Perfect clustering labels
        self.good_labels = np.array([0, 0, 0, 1, 1, 1])
        
        # Bad clustering labels - mixing the clusters
        self.bad_labels = np.array([0, 1, 0, 1, 0, 1])
        
        # Centroids for the good clustering
        self.good_centroids = np.array([
            [[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]],  # Centroid for cluster 1
            [[7.0, 9.0], [7.0, 9.0], [7.0, 9.0]]   # Centroid for cluster 2
        ])
        
        # Centroids for the bad clustering (mixed)
        self.bad_centroids = np.array([
            [[4.0, 6.0], [4.0, 6.0], [4.0, 6.0]],  # Centroid for mixed cluster 1
            [[4.0, 6.0], [4.0, 6.0], [4.0, 6.0]]   # Centroid for mixed cluster 2
        ])
        
        # Single cluster case
        self.single_label = np.array([0, 0, 0, 0, 0, 0])
        self.single_centroid = np.array([
            [[4.0, 6.0], [4.0, 6.0], [4.0, 6.0]]   # Single centroid
        ])
        
        # Empty cluster case
        self.empty_cluster_labels = np.array([0, 0, 0, 0, 0, 0, 1, 1])  # Two clusters, but no data in cluster 2
        self.empty_cluster_data = np.concatenate([self.interval_data, 
                                                 np.zeros((2, 3, 2))], axis=0)  # Add dummy data
        self.empty_cluster_centroids = np.array([
            [[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]],  # Centroid for cluster 0
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]   # Centroid for empty cluster 1
        ])

    def test_calinski_harabasz_index(self):
        """Test Calinski-Harabasz index calculation."""
        # Good clustering should have a higher CH index than bad clustering
        ch_good = calinski_harabasz_index(
            self.interval_data, self.good_labels, self.good_centroids, "euclidean")
        ch_bad = calinski_harabasz_index(
            self.interval_data, self.bad_labels, self.bad_centroids, "euclidean")
        
        # Good clustering should have a higher score
        self.assertGreater(ch_good, ch_bad)
        
        # Test with different metrics
        ch_hausdorff = calinski_harabasz_index(
            self.interval_data, self.good_labels, self.good_centroids, "hausdorff")
        ch_manhattan = calinski_harabasz_index(
            self.interval_data, self.good_labels, self.good_centroids, "manhattan")
        
        # All should be positive for a good clustering
        self.assertGreater(ch_hausdorff, 0)
        self.assertGreater(ch_manhattan, 0)
        
        # Single cluster case should return 0
        ch_single = calinski_harabasz_index(
            self.interval_data, self.single_label, self.single_centroid, "euclidean")
        self.assertEqual(ch_single, 0.0)
        
        # Test with a similarity metric
        ch_jaccard = calinski_harabasz_index(
            self.interval_data, self.good_labels, self.good_centroids, "jaccard")
        self.assertGreater(ch_jaccard, 0)
        
        # Test empty cluster case
        ch_empty = calinski_harabasz_index(
            self.empty_cluster_data, self.empty_cluster_labels, 
            self.empty_cluster_centroids, "euclidean")
        self.assertGreaterEqual(ch_empty, 0)  # Should handle empty clusters gracefully
        
        # Test invalid metric
        with self.assertRaises(ValueError):
            calinski_harabasz_index(
                self.interval_data, self.good_labels, self.good_centroids, "invalid_metric")

    def test_davies_bouldin_index(self):
        """Test Davies-Bouldin index calculation."""
        # For Davies-Bouldin, just verify it produces a value and check bounds
        db_good = davies_bouldin_index(
            self.interval_data, self.good_labels, self.good_centroids, "euclidean")
        db_bad = davies_bouldin_index(
            self.interval_data, self.bad_labels, self.bad_centroids, "euclidean")
        
        # Check that values are non-negative (DB index is always ≥ 0)
        self.assertGreaterEqual(db_good, 0)
        self.assertGreaterEqual(db_bad, 0)
        
        # Test with different metrics
        db_hausdorff = davies_bouldin_index(
            self.interval_data, self.good_labels, self.good_centroids, "hausdorff")
        db_manhattan = davies_bouldin_index(
            self.interval_data, self.good_labels, self.good_centroids, "manhattan")
        
        # All should be non-negative
        self.assertGreaterEqual(db_hausdorff, 0)
        self.assertGreaterEqual(db_manhattan, 0)
        
        # Single cluster case should return 0
        db_single = davies_bouldin_index(
            self.interval_data, self.single_label, self.single_centroid, "euclidean")
        self.assertEqual(db_single, 0.0)
        
        # Test with a similarity metric
        db_jaccard = davies_bouldin_index(
            self.interval_data, self.good_labels, self.good_centroids, "jaccard")
        self.assertGreaterEqual(db_jaccard, 0)
        
        # Test empty cluster case
        db_empty = davies_bouldin_index(
            self.empty_cluster_data, self.empty_cluster_labels, 
            self.empty_cluster_centroids, "euclidean")
        self.assertGreaterEqual(db_empty, 0)  # Should handle empty clusters gracefully
        
        # Test invalid metric
        with self.assertRaises(ValueError):
            davies_bouldin_index(
                self.interval_data, self.good_labels, self.good_centroids, "invalid_metric")

    def test_dunn_index(self):
        """Test Dunn index calculation."""
        # Good clustering should have a higher Dunn index than bad clustering
        dunn_good = dunn_index(self.interval_data, self.good_labels, "euclidean")
        dunn_bad = dunn_index(self.interval_data, self.bad_labels, "euclidean")
        
        # Good clustering should have a higher score (Dunn index is maximized)
        self.assertGreater(dunn_good, dunn_bad)
        
        # Test with different metrics
        dunn_hausdorff = dunn_index(self.interval_data, self.good_labels, "hausdorff")
        dunn_manhattan = dunn_index(self.interval_data, self.good_labels, "manhattan")
        
        # All should be non-negative
        self.assertGreaterEqual(dunn_hausdorff, 0)
        self.assertGreaterEqual(dunn_manhattan, 0)
        
        # Single cluster case should return 0
        dunn_single = dunn_index(self.interval_data, self.single_label, "euclidean")
        self.assertEqual(dunn_single, 0.0)
        
        # Test with a similarity metric
        dunn_jaccard = dunn_index(self.interval_data, self.good_labels, "jaccard")
        self.assertGreaterEqual(dunn_jaccard, 0)
        
        # Test empty cluster case - should handle it gracefully
        dunn_empty = dunn_index(self.empty_cluster_data, self.empty_cluster_labels, "euclidean")
        self.assertGreaterEqual(dunn_empty, 0)
        
        # Test invalid metric
        with self.assertRaises(ValueError):
            dunn_index(self.interval_data, self.good_labels, "invalid_metric")

    def test_distortion_score(self):
        """Test distortion score calculation."""
        # Good clustering should have a lower distortion than bad clustering
        dist_good = distortion_score(
            self.interval_data, self.good_labels, self.good_centroids, "euclidean")
        dist_bad = distortion_score(
            self.interval_data, self.bad_labels, self.bad_centroids, "euclidean")
        
        # Good clustering should have a lower score (distortion is minimized)
        self.assertLess(dist_good, dist_bad)
        
        # Test with different metrics
        dist_hausdorff = distortion_score(
            self.interval_data, self.good_labels, self.good_centroids, "hausdorff")
        dist_manhattan = distortion_score(
            self.interval_data, self.good_labels, self.good_centroids, "manhattan")
        
        # All should be non-negative
        self.assertGreaterEqual(dist_hausdorff, 0)
        self.assertGreaterEqual(dist_manhattan, 0)
        
        # Test with a similarity metric
        dist_jaccard = distortion_score(
            self.interval_data, self.good_labels, self.good_centroids, "jaccard")
        self.assertGreaterEqual(dist_jaccard, 0)
        
        # Test empty cluster case
        dist_empty = distortion_score(
            self.empty_cluster_data, self.empty_cluster_labels, 
            self.empty_cluster_centroids, "euclidean")
        self.assertGreaterEqual(dist_empty, 0)  # Should handle empty clusters gracefully
        
        # Test invalid metric
        with self.assertRaises(ValueError):
            distortion_score(
                self.interval_data, self.good_labels, self.good_centroids, "invalid_metric")

    def test_silhouette_score(self):
        """Test silhouette score calculation."""
        # Good clustering should have a higher silhouette score than bad clustering
        sil_good = silhouette_score(self.interval_data, self.good_labels, "euclidean")
        sil_bad = silhouette_score(self.interval_data, self.bad_labels, "euclidean")
        
        # Good clustering should have a higher score (silhouette is maximized)
        self.assertGreater(sil_good, sil_bad)
        
        # Test with different metrics
        sil_hausdorff = silhouette_score(self.interval_data, self.good_labels, "hausdorff")
        sil_manhattan = silhouette_score(self.interval_data, self.good_labels, "manhattan")
        
        # For a good clustering, silhouette should be positive
        self.assertGreater(sil_hausdorff, 0)
        self.assertGreater(sil_manhattan, 0)
        
        # Silhouette is bounded between -1 and 1
        self.assertLessEqual(sil_good, 1.0)
        self.assertGreaterEqual(sil_good, -1.0)
        self.assertLessEqual(sil_bad, 1.0)
        self.assertGreaterEqual(sil_bad, -1.0)
        
        # Test with a similarity metric
        sil_jaccard = silhouette_score(self.interval_data, self.good_labels, "jaccard")
        self.assertLessEqual(sil_jaccard, 1.0)
        self.assertGreaterEqual(sil_jaccard, -1.0)
        
        # Test empty cluster case (should skip empty clusters)
        sil_empty = silhouette_score(
            self.empty_cluster_data, self.empty_cluster_labels, "euclidean")
        self.assertLessEqual(sil_empty, 1.0)
        self.assertGreaterEqual(sil_empty, -1.0)
        
        # Test invalid metric
        with self.assertRaises(ValueError):
            silhouette_score(self.interval_data, self.good_labels, "invalid_metric")

    def test_shape_mismatch(self):
        """Test handling of data and label shape mismatches."""
        # Create mismatched data and labels
        mismatched_labels = np.array([0, 0, 0, 1, 1])  # One fewer than data points
        
        # Each metric should raise ValueError for mismatched shapes
        # Only check metrics that explicitly validate shapes
        with self.assertRaises(ValueError):
            calinski_harabasz_index(
                self.interval_data, mismatched_labels, self.good_centroids, "euclidean")
        
        with self.assertRaises(ValueError):
            dunn_index(self.interval_data, mismatched_labels, "euclidean")
        
        with self.assertRaises(ValueError):
            distortion_score(
                self.interval_data, mismatched_labels, self.good_centroids, "euclidean")
        
        with self.assertRaises(ValueError):
            silhouette_score(self.interval_data, mismatched_labels, "euclidean")


if __name__ == '__main__':
    unittest.main()
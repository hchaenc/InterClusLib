import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class AbstractIntervalVisualization(ABC):
    """
    Abstract base class for interval data visualization.
    
    This base class defines the common interface that all interval visualization classes
    should implement, including IntervalRadarChart, Interval2d, Interval3d, 
    IntervalParallelCoordinates, and Dendrogram.
    """
    
    @classmethod
    @abstractmethod
    def visualize(cls, intervals=None, centroids=None, labels=None, max_samples_per_cluster=None,
                  figsize=None, title=None, feature_names=None, alpha=None, centroid_alpha=None, 
                  margin=None, **kwargs):
        """
        Abstract method to visualize interval data.
        
        Parameters:
        :param intervals: Interval data with shape (n_samples, n_dims, 2), the last dimension represents [lower, upper]
                         Can be None if only plotting centroids
        :param centroids: Centroid data with shape (n_clusters, n_dims, 2), the last dimension represents [lower, upper]
                         Can be None if only plotting intervals
        :param labels: Optional, array of shape (n_samples,) for cluster labels or categories
        :param max_samples_per_cluster: Maximum number of samples to display per cluster, default is None (show all samples)
        :param figsize: Figure size as tuple (width, height)
        :param title: Figure title
        :param feature_names: List of feature names, default is None, will auto-generate names
        :param alpha: Transparency for regular intervals
        :param centroid_alpha: Transparency for centroids
        :param margin: Margin around axis limits
        :param kwargs: Other visualization-specific parameters
        :return: fig, ax - matplotlib figure and axes objects
        """
        pass
    
    @staticmethod
    def validate_intervals(intervals, n_dims_required=None):
        """
        Validate the format and dimensions of interval data.
        
        Parameters:
        :param intervals: Interval data to validate
        :param n_dims_required: If not None, validate that n_dims equals this value
        :return: None
        :raises: ValueError if data format is incorrect
        """
        if intervals is None:
            return
            
        if intervals.ndim != 3 or intervals.shape[2] != 2:
            raise ValueError(
                f"Interval data should have shape (n_samples, n_dims, 2). "
                f"Got {intervals.shape} instead."
            )
        
        if n_dims_required is not None and intervals.shape[1] != n_dims_required:
            raise ValueError(
                f"This visualization requires interval data with {n_dims_required} dimensions. "
                f"Got {intervals.shape[1]} dimensions instead."
            )
    
    @staticmethod
    def validate_centroids(centroids, n_dims_required=None):
        """
        Validate the format and dimensions of centroid data.
        
        Parameters:
        :param centroids: Centroid data to validate
        :param n_dims_required: If not None, validate that n_dims equals this value
        :return: None
        :raises: ValueError if data format is incorrect
        """
        if centroids is None:
            return
            
        if centroids.ndim != 3 or centroids.shape[2] != 2:
            raise ValueError(
                f"Centroid data should have shape (n_clusters, n_dims, 2). "
                f"Got {centroids.shape} instead."
            )
        
        if n_dims_required is not None and centroids.shape[1] != n_dims_required:
            raise ValueError(
                f"This visualization requires centroid data with {n_dims_required} dimensions. "
                f"Got {centroids.shape[1]} dimensions instead."
            )
    
    @staticmethod
    def setup_cluster_info(intervals, labels, centroids):
        """
        Set up cluster information, including labels, unique labels, and number of clusters.
        
        Parameters:
        :param intervals: Interval data
        :param labels: Cluster labels
        :param centroids: Centroid data
        :return: labels, unique_labels, n_clusters
        """
        if intervals is not None:
            n_samples = intervals.shape[0]
            if labels is None:
                labels = np.zeros(n_samples, dtype=int)
            
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
        elif centroids is not None:
            n_clusters = centroids.shape[0]
            unique_labels = np.arange(n_clusters)
            labels = np.array([]) if labels is None else labels
        else:
            raise ValueError("At least one of 'intervals' or 'centroids' must be provided")
        
        return labels, unique_labels, n_clusters
    
    @staticmethod
    def generate_feature_names(n_features, feature_names=None, prefix="Feature_"):
        """
        Generate feature names.
        
        Parameters:
        :param n_features: Number of features
        :param feature_names: Existing list of feature names, can be None
        :param prefix: Prefix for auto-generated feature names
        :return: List of feature names
        """
        if feature_names is None:
            return [f"{prefix}{i+1}" for i in range(n_features)]
        elif len(feature_names) < n_features:
            return list(feature_names) + [f"{prefix}{i+1}" for i in range(len(feature_names), n_features)]
        return feature_names
    
    @staticmethod
    def generate_cluster_colors(n_clusters, cmap_name='tab10'):
        """
        Generate distinct colors for clusters.
        
        Parameters:
        :param n_clusters: Number of clusters
        :param cmap_name: Colormap name
        :return: List of colors
        """
        colors = plt.cm.get_cmap(cmap_name, n_clusters)
        return [colors(i) for i in range(n_clusters)]
    
    @staticmethod
    def get_feature_boundaries(intervals, centroids, margin=0.1):
        """
        Calculate min and max values for each feature.
        
        Parameters:
        :param intervals: Interval data
        :param centroids: Centroid data
        :param margin: Margin proportion
        :return: feature_mins, feature_maxs
        """
        if intervals is not None:
            # Extract lower and upper bounds
            data_lower = intervals[:, :, 0]
            data_upper = intervals[:, :, 1]
            
            # Initial min/max from data - convert to float explicitly
            feature_mins = np.min(data_lower, axis=0).astype(float)
            feature_maxs = np.max(data_upper, axis=0).astype(float)
        else:
            # Initialize with centroids if no data
            feature_mins = np.min(centroids[:, :, 0], axis=0).astype(float)
            feature_maxs = np.max(centroids[:, :, 1], axis=0).astype(float)
            
        # Update min/max with centroids if available
        if centroids is not None and intervals is not None:
            centroid_mins = np.min(centroids[:, :, 0], axis=0)
            centroid_maxs = np.max(centroids[:, :, 1], axis=0)
            feature_mins = np.minimum(feature_mins, centroid_mins)
            feature_maxs = np.maximum(feature_maxs, centroid_maxs)
        
        # Calculate range for each feature and add margin
        feature_ranges = feature_maxs - feature_mins
        feature_ranges[feature_ranges == 0] = 1.0  # Avoid division by zero for constant features
        
        margin_values = feature_ranges * margin
        feature_mins -= margin_values
        feature_maxs += margin_values
        
        return feature_mins, feature_maxs
    
    @staticmethod
    def scale_to_unit(values, feature_idx, feature_mins, feature_maxs):
        """
        Scale values for a specific feature to 0-1 range for visualization.
        
        Parameters:
        :param values: Values to scale
        :param feature_idx: Feature index
        :param feature_mins: Array of feature minimum values
        :param feature_maxs: Array of feature maximum values
        :return: Scaled values
        """
        min_val = feature_mins[feature_idx]
        max_val = feature_maxs[feature_idx]
        if max_val > min_val:  # Avoid division by zero
            return (values - min_val) / (max_val - min_val)
        else:
            return np.ones_like(values) * 0.5  # Default to middle if min==max
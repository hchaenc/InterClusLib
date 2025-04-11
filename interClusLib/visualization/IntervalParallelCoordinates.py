import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches
import matplotlib as mpl
from interClusLib.visualization.IntervalVisualization import IntervalVisualization

class IntervalParallelCoordinates(IntervalVisualization):
    """Class for parallel coordinates visualization of interval data"""

    @classmethod
    def visualize(cls, intervals=None, centroids=None, labels=None, max_samples_per_cluster=None,
                  figsize=(12, 8), title="Parallel Coordinates", feature_names=None, 
                  alpha=1/6, centroid_alpha=0.4, margin=0.1, beta=0.8, uncertainty_alpha=0.2, 
                  use_bundling=True, **kwargs):
        """
        Unified visualization function for parallel coordinates with curve bundling for interval data
        
        Parameters:
        :param intervals: Interval data with shape (n_samples, n_dims, 2), can be None if only plotting centroids
                         The last dimension represents [lower, upper]
        :param centroids: Centroid data with shape (n_clusters, n_dims, 2), can be None if only plotting intervals
                         The last dimension represents [lower, upper]
        :param labels: Optional, array of shape (n_samples,) for cluster labels or categories
        :param max_samples_per_cluster: Maximum number of samples to display per cluster, default is None (show all samples)
        :param figsize: Figure size, default is (12, 8)
        :param title: Figure title
        :param feature_names: List of feature names, default is None, will auto-generate names
        :param alpha: Smoothness scale (0-0.25), default is 1/6
        :param centroid_alpha: Transparency for centroids, default is 0.4
        :param margin: Margin around axis limits, default is 0.1
        :param beta: Bundling strength (0-1), default is 0.8
        :param uncertainty_alpha: Transparency of uncertainty regions, default is 0.2
        :param use_bundling: Whether to use curve bundling, default is True
        :param kwargs: Other visualization-specific parameters
        :return: fig, ax - matplotlib figure and axes objects
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Validate input: at least one of intervals or centroids must be provided
        if intervals is None and centroids is None:
            raise ValueError("At least one of 'intervals' or 'centroids' must be provided")
            
        # Validate data format if provided
        if intervals is not None:
            cls.validate_intervals(intervals)
            n_samples, n_features = intervals.shape[0], intervals.shape[1]
        else:  # Use centroids to determine n_features
            cls.validate_centroids(centroids)
            n_features = centroids.shape[1]
            n_samples = 0
        
        # Generate feature names
        feature_names = cls.generate_feature_names(n_features, feature_names, prefix="Feature ")
        
        # Set up axes - dynamically adapt to number of dimensions
        axes_x = np.linspace(0, 1, n_features)
        
        # Get feature boundaries for normalization
        all_mins, all_maxs = cls.get_feature_boundaries(intervals, centroids, margin=0)
        
        # Handle potential division by zero
        range_values = all_maxs - all_mins
        range_values[range_values == 0] = 1  # Avoid division by zero
        
        # Extract and normalize data if present
        if intervals is not None:
            data_lower = intervals[:, :, 0]
            data_upper = intervals[:, :, 1]
            data_center = (data_lower + data_upper) / 2
            
            norm_data_lower = (data_lower - all_mins) / range_values
            norm_data_upper = (data_upper - all_mins) / range_values
            norm_data_center = (data_center - all_mins) / range_values
        
        # Draw the axes with better spacing
        for i, x in enumerate(axes_x):
            ax.axvline(x=x, ymin=0, ymax=1, color='black', alpha=0.5)
            ax.text(x, 1.07, feature_names[i], ha='center', va='bottom', fontsize=10, fontweight='bold')
        
            # Format min and max values
            min_val = all_mins[i]
            max_val = all_maxs[i]
            
            # Determine appropriate formatting
            if abs(min_val) < 0.1 or abs(min_val) > 1000:
                min_format = "{:.2e}"
            elif min_val == int(min_val):
                min_format = "{:.0f}"
            else:
                min_format = "{:.1f}"
                
            if abs(max_val) < 0.1 or abs(max_val) > 1000:
                max_format = "{:.2e}"
            elif max_val == int(max_val):
                max_format = "{:.0f}"
            else:
                max_format = "{:.1f}"
            
            # Add value labels with better spacing
            ax.text(x, -0.07, min_format.format(min_val), ha='center', va='top', fontsize=9)
            ax.text(x, 1.0, max_format.format(max_val), ha='center', va='bottom', fontsize=9)
        
        # Set up cluster information
        labels, unique_labels, n_clusters = cls.setup_cluster_info(intervals, labels, centroids)
            
        # Generate colors for clusters
        cluster_colors = cls.generate_cluster_colors(n_clusters, 'tab10')
        
        # Process centroids if provided
        if centroids is not None:
            if centroids.shape[1] != n_features:
                raise ValueError(f"centroids should have shape (n_clusters, {n_features}, 2), got {centroids.shape}")
            
            # Normalize centroids
            centroid_lower = centroids[:, :, 0]
            centroid_upper = centroids[:, :, 1]
            centroid_center = (centroid_lower + centroid_upper) / 2
            
            norm_centroid_lower = (centroid_lower - all_mins) / range_values
            norm_centroid_upper = (centroid_upper - all_mins) / range_values
            norm_centroid_center = (centroid_center - all_mins) / range_values
        
        # Calculate bundling centroids for each segment between axes
        bundling_centroids = []
        if use_bundling:
            for i in range(n_features - 1):
                cluster_bundling_centroids = {}
                
                # If data is available, use it to compute bundling centroids
                if intervals is not None:
                    for idx, lab in enumerate(unique_labels):
                        cluster_mask = labels == lab
                        
                        # Skip clusters with no data points
                        if not np.any(cluster_mask):
                            continue
                        
                        lefts = np.column_stack([
                            np.full(np.sum(cluster_mask), axes_x[i]), 
                            norm_data_center[cluster_mask, i]
                        ])
                        rights = np.column_stack([
                            np.full(np.sum(cluster_mask), axes_x[i+1]), 
                            norm_data_center[cluster_mask, i+1]
                        ])
                        
                        midpoints = []
                        for j in range(lefts.shape[0]):
                            mid_x = (lefts[j, 0] + rights[j, 0]) / 2
                            mid_y = (lefts[j, 1] + rights[j, 1]) / 2
                            midpoints.append((mid_x, mid_y))
                        
                        if midpoints:
                            midpoints = np.array(midpoints)
                            centroid = midpoints.mean(axis=0)
                            cluster_bundling_centroids[lab] = centroid
                # If only centroids are available, use them for bundling centroids
                elif centroids is not None:
                    for idx, lab in enumerate(unique_labels):
                        p_left = np.array([axes_x[i], norm_centroid_center[idx, i]])
                        p_right = np.array([axes_x[i+1], norm_centroid_center[idx, i+1]])
                        mid_x = (p_left[0] + p_right[0]) / 2
                        mid_y = (p_left[1] + p_right[1]) / 2
                        cluster_bundling_centroids[lab] = np.array([mid_x, mid_y])
                
                # Redistribute bundling centroids
                if cluster_bundling_centroids:
                    sorted_centroids = sorted(cluster_bundling_centroids.items(), key=lambda x: x[1][1])
                    y_values = np.linspace(0.1, 0.9, len(sorted_centroids))
                    for (cluster_id, _), y in zip(sorted_centroids, y_values):
                        x = (axes_x[i] + axes_x[i+1]) / 2
                        cluster_bundling_centroids[cluster_id] = np.array([x, y])
                
                bundling_centroids.append(cluster_bundling_centroids)
        
        # Draw data lines and uncertainty regions if data is provided
        legend_handles = []
        if intervals is not None:
            # Check if we're in the case where there are no labels and no centroids
            no_labels_or_centroids = (labels is None or np.all(labels == 0)) and centroids is None
            
            for idx, lab in enumerate(unique_labels):
                cluster_mask = labels == lab
                cluster_indices = np.where(cluster_mask)[0]
                
                # Skip empty clusters
                if len(cluster_indices) == 0:
                    continue
                
                # Limit samples per cluster if specified
                if max_samples_per_cluster is not None and len(cluster_indices) > max_samples_per_cluster:
                    np.random.seed(42)  # For reproducibility
                    cluster_indices = np.random.choice(cluster_indices, max_samples_per_cluster, replace=False)
                
                color = cluster_colors[idx % len(cluster_colors)]
                
                # Draw each interval in the cluster
                for j in cluster_indices:
                    line_data_lower = norm_data_lower[j]
                    line_data_upper = norm_data_upper[j]
                    line_data_center = norm_data_center[j]
                    
                    points_lower = np.column_stack([axes_x, line_data_lower])
                    points_upper = np.column_stack([axes_x, line_data_upper])
                    points_center = np.column_stack([axes_x, line_data_center])
                    
                    # For each pair of adjacent axes, draw curves
                    for i in range(n_features - 1):
                        if use_bundling and i < len(bundling_centroids):
                            centroid = bundling_centroids[i].get(lab, None)
                        else:
                            centroid = None
                        
                        # Center line control points
                        p_left_center = points_center[i]
                        p_right_center = points_center[i+1]
                        
                        (mid_x_center, mid_y_center), (cp1_x, cp1_y), (cp2_x, cp2_y), \
                            (cp3_x, cp3_y), (cp4_x, cp4_y) = \
                            cls._compute_control_points(p_left_center, p_right_center, alpha, beta, centroid)
                        
                        # Lower bound control points
                        p_left_lower = points_lower[i]
                        p_right_lower = points_lower[i+1]
                        
                        (mid_x_lower, mid_y_lower), (cp1_x_lower, cp1_y_lower), (cp2_x_lower, cp2_y_lower), \
                            (cp3_x_lower, cp3_y_lower), (cp4_x_lower, cp4_y_lower) = \
                            cls._compute_control_points(p_left_lower, p_right_lower, alpha, beta, centroid)
                        
                        # Upper bound control points
                        p_left_upper = points_upper[i]
                        p_right_upper = points_upper[i+1]
                        
                        (mid_x_upper, mid_y_upper), (cp1_x_upper, cp1_y_upper), (cp2_x_upper, cp2_y_upper), \
                            (cp3_x_upper, cp3_y_upper), (cp4_x_upper, cp4_y_upper) = \
                            cls._compute_control_points(p_left_upper, p_right_upper, alpha, beta, centroid)
                        
                        # Generate curves
                        # Center curve (left to mid)
                        curve1_center = cls._generate_bezier_curve(
                            p_left_center, 
                            np.array([cp1_x, cp1_y]), 
                            np.array([cp2_x, cp2_y]), 
                            np.array([mid_x_center, mid_y_center])
                        )
                        
                        # Center curve (mid to right)
                        curve2_center = cls._generate_bezier_curve(
                            np.array([mid_x_center, mid_y_center]),
                            np.array([cp3_x, cp3_y]),
                            np.array([cp4_x, cp4_y]),
                            p_right_center
                        )
                        
                        curve_center = np.vstack([curve1_center, curve2_center])
                        
                        # Lower bound curve (left to mid)
                        curve1_lower = cls._generate_bezier_curve(
                            p_left_lower, 
                            np.array([cp1_x_lower, cp1_y_lower]), 
                            np.array([cp2_x_lower, cp2_y_lower]), 
                            np.array([mid_x_lower, mid_y_lower])
                        )
                        
                        # Lower bound curve (mid to right)
                        curve2_lower = cls._generate_bezier_curve(
                            np.array([mid_x_lower, mid_y_lower]),
                            np.array([cp3_x_lower, cp3_y_lower]),
                            np.array([cp4_x_lower, cp4_y_lower]),
                            p_right_lower
                        )
                        
                        curve_lower = np.vstack([curve1_lower, curve2_lower])
                        
                        # Upper bound curve (left to mid)
                        curve1_upper = cls._generate_bezier_curve(
                            p_left_upper, 
                            np.array([cp1_x_upper, cp1_y_upper]), 
                            np.array([cp2_x_upper, cp2_y_upper]), 
                            np.array([mid_x_upper, mid_y_upper])
                        )
                        
                        # Upper bound curve (mid to right)
                        curve2_upper = cls._generate_bezier_curve(
                            np.array([mid_x_upper, mid_y_upper]),
                            np.array([cp3_x_upper, cp3_y_upper]),
                            np.array([cp4_x_upper, cp4_y_upper]),
                            p_right_upper
                        )
                        
                        curve_upper = np.vstack([curve1_upper, curve2_upper])
                        
                        # Draw uncertainty region
                        verts = np.vstack([curve_lower, curve_upper[::-1]])
                        poly = mpatches.Polygon(verts, closed=True, facecolor=color, alpha=uncertainty_alpha)
                        ax.add_patch(poly)
                        
                        # Draw center line with deeper color when no labels or centroids are provided
                        if no_labels_or_centroids:
                            # Create a darker version of the color for the center line
                            darker_color = np.array(color)
                            darker_color[:3] = darker_color[:3] * 0.7  # Make it 40% darker
                            ax.plot(curve_center[:, 0], curve_center[:, 1], color=darker_color, alpha=1.0, linewidth=1.5)
                        else:
                            # Use original color with lower alpha
                            ax.plot(curve_center[:, 0], curve_center[:, 1], color=color, alpha=0.7, linewidth=1)
                
                # Add legend entry for this cluster (only once per cluster)
                if n_clusters > 1:
                    legend_handle = plt.Line2D(
                        [0], [0], color=color, lw=2, marker='_', 
                        markersize=0, markerfacecolor=color, markeredgecolor='none',
                        label=f'Cluster {lab+1}'
                    )
                    legend_handles.append(legend_handle)
        
        # Draw centroids if provided
        centroid_legend_handles = []
        if centroids is not None:
            for idx, lab in enumerate(unique_labels[:n_clusters]):
                # Skip this centroid if it corresponds to a cluster with no data points
                # (only matters when both data and centroids are provided)
                if intervals is not None and np.sum(labels == lab) == 0 and idx < len(labels):
                    continue
                
                # Use a darker, more saturated version of the cluster color
                base_color = np.array(cluster_colors[idx % len(cluster_colors)])
                # Make color darker for centroid
                dark_color = np.clip(base_color * 0.7, 0, 1)
                dark_color[3] = 1.0  # Full opacity
                
                # Draw interval region for centroid
                for j in range(n_features - 1):
                    # Get points for the centroids
                    p_left_lower = np.array([axes_x[j], norm_centroid_lower[idx, j]])
                    p_right_lower = np.array([axes_x[j+1], norm_centroid_lower[idx, j+1]])
                    p_left_upper = np.array([axes_x[j], norm_centroid_upper[idx, j]])
                    p_right_upper = np.array([axes_x[j+1], norm_centroid_upper[idx, j+1]])
                    p_left_center = np.array([axes_x[j], norm_centroid_center[idx, j]])
                    p_right_center = np.array([axes_x[j+1], norm_centroid_center[idx, j+1]])
                    
                    # Get centroid point for bundling
                    bundle_centroid = None
                    if use_bundling and j < len(bundling_centroids):
                        bundle_centroid = bundling_centroids[j].get(lab, None)
                    
                    # Calculate control points
                    (mid_x_center, mid_y_center), (cp1_x, cp1_y), (cp2_x, cp2_y), \
                        (cp3_x, cp3_y), (cp4_x, cp4_y) = \
                        cls._compute_control_points(p_left_center, p_right_center, 
                                                   alpha, beta, bundle_centroid)
                    
                    (mid_x_lower, mid_y_lower), (cp1_x_lower, cp1_y_lower), (cp2_x_lower, cp2_y_lower), \
                        (cp3_x_lower, cp3_y_lower), (cp4_x_lower, cp4_y_lower) = \
                        cls._compute_control_points(p_left_lower, p_right_lower, 
                                                   alpha, beta, bundle_centroid)
                    
                    (mid_x_upper, mid_y_upper), (cp1_x_upper, cp1_y_upper), (cp2_x_upper, cp2_y_upper), \
                        (cp3_x_upper, cp3_y_upper), (cp4_x_upper, cp4_y_upper) = \
                        cls._compute_control_points(p_left_upper, p_right_upper, 
                                                   alpha, beta, bundle_centroid)
                    
                    # Generate curves
                    curve1_center = cls._generate_bezier_curve(
                        p_left_center, 
                        np.array([cp1_x, cp1_y]), 
                        np.array([cp2_x, cp2_y]), 
                        np.array([mid_x_center, mid_y_center])
                    )
                    
                    curve2_center = cls._generate_bezier_curve(
                        np.array([mid_x_center, mid_y_center]),
                        np.array([cp3_x, cp3_y]),
                        np.array([cp4_x, cp4_y]),
                        p_right_center
                    )
                    
                    curve_center = np.vstack([curve1_center, curve2_center])
                    
                    curve1_lower = cls._generate_bezier_curve(
                        p_left_lower, 
                        np.array([cp1_x_lower, cp1_y_lower]), 
                        np.array([cp2_x_lower, cp2_y_lower]), 
                        np.array([mid_x_lower, mid_y_lower])
                    )
                    
                    curve2_lower = cls._generate_bezier_curve(
                        np.array([mid_x_lower, mid_y_lower]),
                        np.array([cp3_x_lower, cp3_y_lower]),
                        np.array([cp4_x_lower, cp4_y_lower]),
                        p_right_lower
                    )
                    
                    curve_lower = np.vstack([curve1_lower, curve2_lower])
                    
                    curve1_upper = cls._generate_bezier_curve(
                        p_left_upper, 
                        np.array([cp1_x_upper, cp1_y_upper]), 
                        np.array([cp2_x_upper, cp2_y_upper]), 
                        np.array([mid_x_upper, mid_y_upper])
                    )
                    
                    curve2_upper = cls._generate_bezier_curve(
                        np.array([mid_x_upper, mid_y_upper]),
                        np.array([cp3_x_upper, cp3_y_upper]),
                        np.array([cp4_x_upper, cp4_y_upper]),
                        p_right_upper
                    )
                    
                    curve_upper = np.vstack([curve1_upper, curve2_upper])
                    
                    # Draw uncertainty region with higher alpha
                    verts = np.vstack([curve_lower, curve_upper[::-1]])
                    poly = mpatches.Polygon(verts, closed=True, facecolor=dark_color, 
                                          alpha=centroid_alpha, linewidth=0)
                    ax.add_patch(poly)
                    
                    # Draw center line with thicker line
                    ax.plot(curve_center[:, 0], curve_center[:, 1], color=dark_color, 
                          alpha=1.0, linewidth=2.5, zorder=10)
                
                # Draw markers at centroid points on each axis
                for j in range(n_features):
                    ax.plot(axes_x[j], norm_centroid_center[idx, j], 'o', 
                          color=dark_color, markersize=8, markeredgecolor='white', 
                          markeredgewidth=1.0, zorder=11)
                
                # Add legend entry for this centroid
                if n_clusters > 1:
                    centroid_handle = plt.Line2D(
                        [0], [0], color=dark_color, lw=3, 
                        marker='o', markersize=6, markeredgecolor='white',
                        markeredgewidth=0.8,
                        label=f'Centroid {lab+1}'
                    )
                    centroid_legend_handles.append(centroid_handle)
        
        # Add all legend handles
        all_handles = legend_handles + centroid_legend_handles
        if all_handles and n_clusters > 1:
            ax.legend(handles=all_handles, loc='upper right', frameon=True, 
                     title="Clusters", title_fontsize=10,
                     fontsize=9, framealpha=0.7)
        
        # Set axis limits with extended padding
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.08, 1.10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set title if provided
        if title:
            ax.set_title(title)
        
        # Apply tight layout
        plt.tight_layout()
        
        return fig, ax
    
    @classmethod
    def _generate_bezier_curve(cls, p0, p1, p2, p3, num_points=50):
        """
        Generate a cubic Bezier curve from 4 control points.
        
        Parameters:
        :param p0: Start point (x, y)
        :param p1: First control point (x, y)
        :param p2: Second control point (x, y)
        :param p3: End point (x, y)
        :param num_points: Number of points in the curve, default is 50
        :return: Array of shape (num_points, 2) containing points on the curve
        """
        t = np.linspace(0, 1, num_points)
        
        # Cubic Bezier formula
        curve = (1-t)**3 * p0[:, np.newaxis] + \
                3 * (1-t)**2 * t * p1[:, np.newaxis] + \
                3 * (1-t) * t**2 * p2[:, np.newaxis] + \
                t**3 * p3[:, np.newaxis]
        
        return curve.T

    @classmethod
    def _compute_control_points(cls, p_left, p_right, alpha=1/6, beta=0.8, centroid=None):
        """
        Compute the control points for a cubic Bezier curve between two points
        with proper C1 continuity.
        
        Parameters:
        :param p_left: Point at the left axis (x, y)
        :param p_right: Point at the right axis (x, y)
        :param alpha: Smoothness scale (0-0.25), default is 1/6
        :param beta: Bundling strength (0-1), default is 0.8
        :param centroid: Cluster centroid position at the virtual axis (x, y), optional
        :return: Tuple of 5 points: (mid_point, cp1, cp2, cp3, cp4)
        """
        # Middle point (original polyline intersection with virtual axis)
        mid_x = (p_left[0] + p_right[0]) / 2
        mid_y = (p_left[1] + p_right[1]) / 2
        
        # If we have a centroid for bundling, adjust the mid point
        if centroid is not None:
            # Original midpoint
            q = np.array([mid_x, mid_y])
            # Adjust midpoint towards centroid based on beta
            q_prime = q + beta * (centroid - q)
            mid_x, mid_y = q_prime
        
        # Control points for the left curve segment
        cp1_x = p_left[0] + alpha * (mid_x - p_left[0])
        cp1_y = p_left[1]
        
        # Control points for the right curve segment
        cp2_x = mid_x - alpha * (mid_x - p_left[0])
        cp2_y = mid_y
        
        cp3_x = mid_x + alpha * (p_right[0] - mid_x)
        cp3_y = mid_y
        
        cp4_x = p_right[0] - alpha * (p_right[0] - mid_x)
        cp4_y = p_right[1]
        
        return (mid_x, mid_y), (cp1_x, cp1_y), (cp2_x, cp2_y), (cp3_x, cp3_y), (cp4_x, cp4_y)
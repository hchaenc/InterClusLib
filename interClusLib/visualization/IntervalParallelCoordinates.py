import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class IntervalParallelCoordinates:

    # Functions for generating Bezier curves
    @staticmethod
    def generate_bezier_curve(p0, p1, p2, p3, num_points=50):
        """Generate a cubic Bezier curve from 4 control points."""
        t = np.linspace(0, 1, num_points)
        
        # Cubic Bezier formula
        curve = (1-t)**3 * p0[:, np.newaxis] + \
                3 * (1-t)**2 * t * p1[:, np.newaxis] + \
                3 * (1-t) * t**2 * p2[:, np.newaxis] + \
                t**3 * p3[:, np.newaxis]
        
        return curve.T

    @staticmethod
    def compute_control_points(p_left, p_right, alpha=1/6, beta=0.8, centroid=None):
        """
        Compute the control points for a cubic Bezier curve between two points
        with proper C1 continuity.
        
        Parameters:
        - p_left: point at the left axis
        - p_right: point at the right axis
        - alpha: smoothness scale (0-0.25)
        - beta: bundling strength (0-1)
        - centroid: cluster centroid position at the virtual axis
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

    def plot_interval_curves(data, feature_names=None, clusters=None, alpha=1/6, beta=0.8, ax=None, 
                    use_bundling=True, use_color=True, uncertainty_alpha=0.2):
        """
        Plot parallel coordinates with curve bundling for interval data.
        
        Parameters:
        - data: interval data in format (n_samples, n_dim, 2), where the last dimension represents [lower, upper]
        - feature_names: list of feature names, if None, will auto-generate names
        - clusters: cluster assignments, if None, all data will be treated as one cluster
        - alpha: smoothness scale (0-0.25)
        - beta: bundling strength (0-1)
        - ax: matplotlib axis
        - use_bundling: whether to use curve bundling
        - use_color: whether to use color coding
        - uncertainty_alpha: transparency of uncertainty regions
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        n_samples, n_features = data.shape[0], data.shape[1]
        
        # If feature_names is None, auto-generate feature names
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(n_features)]
        
        # Set up axes - dynamically adapt to number of dimensions
        axes_x = np.linspace(0, 1, n_features)
        
        # Extract lower and upper bounds
        data_lower = data[:, :, 0]
        data_upper = data[:, :, 1]
        data_center = (data_lower + data_upper) / 2
        
        # Normalize data
        all_mins = np.min(data_lower, axis=0)
        all_maxs = np.max(data_upper, axis=0)
        
        # Handle potential division by zero
        range_values = all_maxs - all_mins
        range_values[range_values == 0] = 1  # Avoid division by zero
        
        norm_data_lower = (data_lower - all_mins) / range_values
        norm_data_upper = (data_upper - all_mins) / range_values
        norm_data_center = (data_center - all_mins) / range_values
        
        # Draw the axes
        for i, x in enumerate(axes_x):
            ax.axvline(x=x, ymin=0, ymax=1, color='black', alpha=0.5)
            ax.text(x, 1.02, feature_names[i], ha='center', va='bottom', fontsize=9)
        
        # Set up cluster colors
        if clusters is None:
            clusters = np.zeros(n_samples, dtype=int)
        
        n_clusters = len(np.unique(clusters))
        colors = [plt.cm.get_cmap('tab10')(i/max(1, n_clusters-1)) for i in range(n_clusters)]
        
        # Calculate centroids for bundling
        if use_bundling:
            centroids = []
            for i in range(n_features - 1):
                cluster_centroids = {}
                for cluster_id in np.unique(clusters):
                    cluster_mask = clusters == cluster_id
                    
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
                        cluster_centroids[cluster_id] = centroid
                
                # Redistribute centroids
                if cluster_centroids:
                    sorted_centroids = sorted(cluster_centroids.items(), key=lambda x: x[1][1])
                    y_values = np.linspace(0.1, 0.9, len(sorted_centroids))
                    for (cluster_id, _), y in zip(sorted_centroids, y_values):
                        x = (axes_x[i] + axes_x[i+1]) / 2
                        cluster_centroids[cluster_id] = np.array([x, y])
                
                centroids.append(cluster_centroids)
        
        # Draw each line and its uncertainty region
        for j in range(n_samples):
            line_data_lower = norm_data_lower[j]
            line_data_upper = norm_data_upper[j]
            line_data_center = norm_data_center[j]
            
            points_lower = np.column_stack([axes_x, line_data_lower])
            points_upper = np.column_stack([axes_x, line_data_upper])
            points_center = np.column_stack([axes_x, line_data_center])
            
            if use_color:
                line_color = colors[clusters[j]]
            else:
                line_color = 'steelblue'
            
            # For each pair of adjacent axes, draw curves
            for i in range(n_features - 1):
                if use_bundling:
                    centroid = centroids[i].get(clusters[j], None)
                else:
                    centroid = None
                
                # Center line control points
                p_left_center = points_center[i]
                p_right_center = points_center[i+1]
                
                (mid_x_center, mid_y_center), (cp1_x, cp1_y), (cp2_x, cp2_y), \
                    (cp3_x, cp3_y), (cp4_x, cp4_y) = \
                    IntervalParallelCoordinates.compute_control_points(p_left_center, p_right_center, alpha, beta, centroid)
                
                # Lower bound control points
                p_left_lower = points_lower[i]
                p_right_lower = points_lower[i+1]
                
                (mid_x_lower, mid_y_lower), (cp1_x_lower, cp1_y_lower), (cp2_x_lower, cp2_y_lower), \
                    (cp3_x_lower, cp3_y_lower), (cp4_x_lower, cp4_y_lower) = \
                    IntervalParallelCoordinates.compute_control_points(p_left_lower, p_right_lower, alpha, beta, centroid)
                
                # Upper bound control points
                p_left_upper = points_upper[i]
                p_right_upper = points_upper[i+1]
                
                (mid_x_upper, mid_y_upper), (cp1_x_upper, cp1_y_upper), (cp2_x_upper, cp2_y_upper), \
                    (cp3_x_upper, cp3_y_upper), (cp4_x_upper, cp4_y_upper) = \
                    IntervalParallelCoordinates.compute_control_points(p_left_upper, p_right_upper, alpha, beta, centroid)
                
                # Generate curves
                # Center curve (left to mid)
                curve1_center = IntervalParallelCoordinates.generate_bezier_curve(
                    p_left_center, 
                    np.array([cp1_x, cp1_y]), 
                    np.array([cp2_x, cp2_y]), 
                    np.array([mid_x_center, mid_y_center])
                )
                
                # Center curve (mid to right)
                curve2_center = IntervalParallelCoordinates.generate_bezier_curve(
                    np.array([mid_x_center, mid_y_center]),
                    np.array([cp3_x, cp3_y]),
                    np.array([cp4_x, cp4_y]),
                    p_right_center
                )
                
                curve_center = np.vstack([curve1_center, curve2_center])
                
                # Lower bound curve (left to mid)
                curve1_lower = IntervalParallelCoordinates.generate_bezier_curve(
                    p_left_lower, 
                    np.array([cp1_x_lower, cp1_y_lower]), 
                    np.array([cp2_x_lower, cp2_y_lower]), 
                    np.array([mid_x_lower, mid_y_lower])
                )
                
                # Lower bound curve (mid to right)
                curve2_lower = IntervalParallelCoordinates.generate_bezier_curve(
                    np.array([mid_x_lower, mid_y_lower]),
                    np.array([cp3_x_lower, cp3_y_lower]),
                    np.array([cp4_x_lower, cp4_y_lower]),
                    p_right_lower
                )
                
                curve_lower = np.vstack([curve1_lower, curve2_lower])
                
                # Upper bound curve (left to mid)
                curve1_upper = IntervalParallelCoordinates.generate_bezier_curve(
                    p_left_upper, 
                    np.array([cp1_x_upper, cp1_y_upper]), 
                    np.array([cp2_x_upper, cp2_y_upper]), 
                    np.array([mid_x_upper, mid_y_upper])
                )
                
                # Upper bound curve (mid to right)
                curve2_upper = IntervalParallelCoordinates.generate_bezier_curve(
                    np.array([mid_x_upper, mid_y_upper]),
                    np.array([cp3_x_upper, cp3_y_upper]),
                    np.array([cp4_x_upper, cp4_y_upper]),
                    p_right_upper
                )
                
                curve_upper = np.vstack([curve1_upper, curve2_upper])
                
                # Draw uncertainty region
                verts = np.vstack([curve_lower, curve_upper[::-1]])
                poly = mpatches.Polygon(verts, closed=True, facecolor=line_color, alpha=uncertainty_alpha)
                ax.add_patch(poly)
                
                # Draw center line
                ax.plot(curve_center[:, 0], curve_center[:, 1], color=line_color, alpha=0.7, linewidth=1)
        
        # Set axis limits
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add feature value labels
        for i, x in enumerate(axes_x):
            ax.text(x, 0, f"{all_mins[i]:.1f}", ha='center', va='top', fontsize=8)
            ax.text(x, 1, f"{all_maxs[i]:.1f}", ha='center', va='bottom', fontsize=8)
        
        # Add legend (when using color coding)
        if use_color and n_clusters > 1:
            handles = []
            for i in range(n_clusters):
                handles.append(plt.Line2D([0], [0], color=colors[i], lw=2, label=f'Cluster {i+1}'))
            ax.legend(handles=handles, loc='upper right')
        
        return ax
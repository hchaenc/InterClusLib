import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.colors import to_rgba

class IntervalRadarChart:
    """Class for radar chart visualization of interval data"""

    @staticmethod
    def visualize(intervals=None, centroids=None, labels=None, max_samples_per_cluster=None,
                  figsize=(10, 10), title="Radar Chart Visualization", 
                  feature_names=None, alpha=0.2, centroid_alpha=0.4, margin=0.1):
        """
        Unified visualization function for radar chart with interval data.
        Each feature axis uses its own independent scale.
        
        Parameters:
        :param intervals: Interval data with shape (n_samples, n_dims, 2), can be None if only plotting centroids
                         The last dimension represents [lower, upper]
        :param centroids: Centroid data with shape (n_clusters, n_dims, 2), can be None if only plotting intervals
                         The last dimension represents [lower, upper]
        :param labels: Optional, array of shape (n_samples,) for cluster labels or categories
        :param figsize: Figure size, default is (10, 10)
        :param title: Figure title
        :param feature_names: List of feature names, default is None, will auto-generate names
        :param alpha: Transparency for regular intervals, default is 0.2
        :param centroid_alpha: Transparency for centroids, default is 0.4
        :param margin: Margin around axis limits as fraction of data range, default is 0.1
        :param max_samples_per_cluster: Maximum number of samples to display per cluster, default is None (show all samples)
        :return: fig, ax - matplotlib figure and axes objects
        """
        # Create figure and axis if not provided
        fig = plt.figure(figsize=figsize)
        
        # Create the main radar plot with slightly reduced height to make room for title
        ax = fig.add_subplot(111, polar=True)
        
        # Add padding at the top to create more space between title and plot
        plt.subplots_adjust(top=0.85)
            
        # Ensure either data or centroids (or both) are provided
        if intervals is None and centroids is None:
            raise ValueError("At least one of 'intervals' or 'centroids' must be provided")
            
        # Determine the number of features
        if intervals is not None:
            # Validate intervals shape
            if intervals.ndim != 3 or intervals.shape[2] != 2:
                raise ValueError(
                    f"Expected intervals to have shape (n_samples, n_dims, 2). "
                    f"Got {intervals.shape} instead."
                )
            n_samples, n_features = intervals.shape[0], intervals.shape[1]
        else:
            # Validate centroids shape
            if centroids.ndim != 3 or centroids.shape[2] != 2:
                raise ValueError(
                    f"Expected centroids to have shape (n_clusters, n_dims, 2). "
                    f"Got {centroids.shape} instead."
                )
            n_features = centroids.shape[1]
            n_samples = 0
        
        # Auto-generate feature names if needed
        if feature_names is None:
            feature_names = [f"Feature_{i+1}" for i in range(n_features)]
        elif len(feature_names) < n_features:
            # Extend feature names if fewer than n_features are provided
            feature_names = list(feature_names) + [f"Feature_{i+1}" for i in range(len(feature_names), n_features)]
        
        # Setup the angles for the radar chart (equally spaced)
        angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
        # Close the loop for plotting
        angles.append(angles[0])
        
        # Extract data - using original values without normalization
        if intervals is not None:
            data_lower = intervals[:, :, 0]
            data_upper = intervals[:, :, 1]
            data_center = (data_lower + data_upper) / 2
            
            # Find min/max values per feature
            feature_mins = np.min(data_lower, axis=0)
            feature_maxs = np.max(data_upper, axis=0)
        else:
            feature_mins = np.min(centroids[:, :, 0], axis=0)
            feature_maxs = np.max(centroids[:, :, 1], axis=0)
            
        # Update min/max with centroids if available
        if centroids is not None and intervals is not None:
            centroid_mins = np.min(centroids[:, :, 0], axis=0)
            centroid_maxs = np.max(centroids[:, :, 1], axis=0)
            feature_mins = np.minimum(feature_mins, centroid_mins)
            feature_maxs = np.maximum(feature_maxs, centroid_maxs)
        
        # Calculate range for each feature and add margin
        feature_ranges = feature_maxs - feature_mins
        feature_ranges[feature_ranges == 0] = 1.0  # Avoid division by zero for constant features
        
        # Add margin to the min/max values
        margin_values = feature_ranges * margin
        feature_mins -= margin_values
        feature_maxs += margin_values
        
        # Set up cluster information
        if intervals is not None:
            if labels is None:
                labels = np.zeros(n_samples, dtype=int)
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
        elif centroids is not None:
            n_clusters = centroids.shape[0]
            unique_labels = np.arange(n_clusters)
            # Create a dummy labels array if intervals is None
            labels = np.array([]) if labels is None else labels
            
        # Generate distinct colors for clusters
        colors = plt.cm.get_cmap('tab10', n_clusters)
        cluster_colors = [colors(i) for i in range(n_clusters)]
        
        # Process centroids if provided
        if centroids is not None:
            if centroids.shape[1] != n_features:
                raise ValueError(f"centroids should have shape (n_clusters, {n_features}, 2)")
            
            centroid_lower = centroids[:, :, 0]
            centroid_upper = centroids[:, :, 1]
            centroid_center = (centroid_lower + centroid_upper) / 2
        
        # Remove the default circle border - always hidden now
        ax.spines['polar'].set_visible(False)
        
        # Define a fixed radius scale for plotting (0 to 1)
        r_min = 0
        r_max = 1
        
        # Draw circular grid lines - LIGHTER GRID LINES
        grid_levels = 5
        grid_radii = np.linspace(r_min, r_max, grid_levels + 1)[1:]  # Skip the first (center point)
        
        for i, radius in enumerate(grid_radii):
            # Draw a complete circle for each grid level
            circle = plt.Circle((0, 0), radius, transform=ax.transData._b, 
                             fill=False, edgecolor='gray', alpha=0.2,
                             linestyle='-', linewidth=0.8)
            ax.add_patch(circle)
        
        # Draw radial lines for each feature with scale labels
        for i in range(n_features):
            # Draw radial line from center to edge
            ax.plot([angles[i], angles[i]], [r_min, r_max], 
                 color='gray', alpha=0.4, linestyle='-', linewidth=0.8)
            
            # Calculate positions for the tick marks along this radial line
            # Include both min and max, plus intermediate ticks
            tick_positions = np.linspace(r_min, r_max, grid_levels + 1)
            
            # Calculate corresponding feature values at these positions
            feature_values = feature_mins[i] + (feature_maxs[i] - feature_mins[i]) * tick_positions
            
            # Place tick labels along the radial line (including min value at center)
            for j, (radius, value) in enumerate(zip(tick_positions, feature_values)):
                # Only show min value (at j=0) and max value (j=len-1), plus one intermediate tick
                if j == 0 or j == len(tick_positions)-1 or j == 2:
                    # Format label based on magnitude
                    if abs(value) >= 1000:
                        label = f"{value/1000:.1f}k"
                    elif abs(value) >= 100:
                        label = f"{value:.0f}"
                    elif abs(value) >= 10:
                        label = f"{value:.1f}"
                    else:
                        label = f"{value:.2f}"
                    
                    # Position the label with appropriate offset based on position
                    label_angle = angles[i]
                    
                    # For center (min value), position it slightly away from center point
                    if j == 0:  # Min value at center
                        # Skip the center value to avoid overcrowding
                        continue
                    else:
                        # For other labels, position relative to the grid line
                        dx = 0.03 * np.cos(label_angle)
                        dy = 0.03 * np.sin(label_angle)
                        
                        # Adjust text alignment based on angle position
                        if 0 <= label_angle < np.pi/4 or 7*np.pi/4 <= label_angle <= 2*np.pi:
                            ha, va = 'left', 'center'
                        elif np.pi/4 <= label_angle < 3*np.pi/4:
                            ha, va = 'center', 'bottom'
                        elif 3*np.pi/4 <= label_angle < 5*np.pi/4:
                            ha, va = 'right', 'center'
                        else:
                            ha, va = 'center', 'top'
                        
                        # Make labels larger and more visible with bold font - no background
                        ax.text(label_angle + dx, radius + dy, label, 
                             color='black', fontsize=12, fontweight='bold',
                             ha=ha, va=va, alpha=1.0)
            
            # Place feature name label closer to the chart edge
            angle_rad = angles[i]
            label_distance = r_max + 0.15  # Reduced distance for feature names
            
            # Calculate text alignment based on angle
            if 0 <= angle_rad < np.pi/4 or 7*np.pi/4 <= angle_rad <= 2*np.pi:
                ha, va = 'left', 'center'
            elif np.pi/4 <= angle_rad < 3*np.pi/4:
                ha, va = 'center', 'bottom'
            elif 3*np.pi/4 <= angle_rad < 5*np.pi/4:
                ha, va = 'right', 'center'
            else:
                ha, va = 'center', 'top'
            
            # Place label at the angle with enhanced visibility - no background
            ax.text(angle_rad, label_distance, feature_names[i], 
                 color='black', fontsize=12, fontweight='bold',
                 horizontalalignment=ha, verticalalignment=va)
        
        # Check if we're in the case where there are no labels and no centroids
        no_labels_or_centroids = (labels is None or np.all(labels == 0)) and centroids is None
        
        # Function to normalize values for each feature to the 0-1 scale
        def normalize_values(values):
            normalized = np.zeros_like(values)
            for j in range(n_features):
                min_val = feature_mins[j]
                max_val = feature_maxs[j]
                if max_val > min_val:  # Avoid division by zero
                    normalized[:, j] = (values[:, j] - min_val) / (max_val - min_val)
                else:
                    normalized[:, j] = 0.5  # Default to middle if min==max
            return normalized
        
        # Draw data samples using normalized values
        legend_handles = []
        if intervals is not None:
            # For each cluster, select a limited number of samples to display
            for idx, lab in enumerate(unique_labels):
                cluster_indices = np.where(labels == lab)[0]
                
                # Limit samples per cluster only if max_samples_per_cluster is specified
                if max_samples_per_cluster is not None and len(cluster_indices) > max_samples_per_cluster:
                    np.random.seed(42)  # For reproducibility
                    cluster_indices = np.random.choice(cluster_indices, max_samples_per_cluster, replace=False)
                
                # Get color for this cluster
                line_color = cluster_colors[idx % len(cluster_colors)]
                
                for j in cluster_indices:
                    # Normalize the values for this sample
                    sample_lower = data_lower[j:j+1]
                    sample_upper = data_upper[j:j+1]
                    sample_center = data_center[j:j+1]
                    
                    norm_lower = normalize_values(sample_lower).flatten()
                    norm_upper = normalize_values(sample_upper).flatten()
                    norm_center = normalize_values(sample_center).flatten()
                    
                    # Close the loop for each sample
                    line_data_lower = np.append(norm_lower, norm_lower[0])
                    line_data_upper = np.append(norm_upper, norm_upper[0])
                    line_data_center = np.append(norm_center, norm_center[0])
                    
                    # Plot center line with different opacity based on whether we have labels/centroids
                    if no_labels_or_centroids:
                        # Create a darker version of the color for the center line
                        darker_color = np.array(line_color)
                        darker_color[:3] = darker_color[:3] * 0.7  # Make it 30% darker
                        ax.plot(angles, line_data_center, color=darker_color, 
                              alpha=1.0, linewidth=1.8, zorder=5)  # Higher alpha and thicker line
                    else:
                        # Original approach with regular color but slightly higher opacity
                        ax.plot(angles, line_data_center, color=line_color, 
                              alpha=0.7, linewidth=1.2, zorder=5)
                    
                    # Plot lower and upper bounds with higher opacity
                    ax.plot(angles, line_data_lower, color=line_color, 
                          alpha=0.5, linewidth=0.8, linestyle='--', zorder=4)
                    ax.plot(angles, line_data_upper, color=line_color, 
                          alpha=0.5, linewidth=0.8, linestyle='--', zorder=4)
                    
                    # Fill the area between upper and lower bounds with higher opacity
                    ax.fill_between(angles, line_data_lower, line_data_upper, 
                                 color=line_color, alpha=alpha, zorder=3)
                
                # Add legend entry for this cluster (only once per cluster)
                legend_handle = plt.Line2D(
                    [0], [0], color=line_color, lw=2, 
                    marker='_', markersize=0, markerfacecolor=line_color,
                    label=f'Cluster {lab+1}'
                )
                legend_handles.append(legend_handle)
        
        # Draw centroids with enhanced visibility
        centroid_legend_handles = []
        if centroids is not None:
            for idx, lab in enumerate(unique_labels[:n_clusters]):
                # Skip if no data points in this cluster
                if intervals is not None and np.sum(labels == lab) == 0 and idx < len(labels):
                    continue
                
                # Normalize the centroid values
                cent_lower = centroid_lower[idx:idx+1]
                cent_upper = centroid_upper[idx:idx+1]
                cent_center = centroid_center[idx:idx+1]
                
                norm_cent_lower = normalize_values(cent_lower).flatten()
                norm_cent_upper = normalize_values(cent_upper).flatten()
                norm_cent_center = normalize_values(cent_center).flatten()
                
                # Get a more prominent color for centroids
                base_color = cluster_colors[idx % len(cluster_colors)]
                dark_color = tuple([c*0.7 for c in base_color[:3]] + [1.0])  # Darker, fully opaque
                
                # Close the loop for centroids
                centroid_lower_loop = np.append(norm_cent_lower, norm_cent_lower[0])
                centroid_upper_loop = np.append(norm_cent_upper, norm_cent_upper[0])
                centroid_center_loop = np.append(norm_cent_center, norm_cent_center[0])
                
                # Plot center line for centroid with high visibility - always thicker line now
                line_width = 3.0  # Fixed thickness
                ax.plot(angles, centroid_center_loop, color=dark_color, 
                      alpha=1.0, linewidth=line_width, zorder=8)
                
                # Plot lower and upper bounds with dashed lines - higher opacity
                ax.plot(angles, centroid_lower_loop, color=dark_color, 
                      alpha=0.9, linewidth=1.2, linestyle='--', zorder=7)
                ax.plot(angles, centroid_upper_loop, color=dark_color, 
                      alpha=0.9, linewidth=1.2, linestyle='--', zorder=7)
                
                # Fill the area between upper and lower bounds with higher opacity
                ax.fill_between(angles, centroid_lower_loop, centroid_upper_loop, 
                             color=dark_color, alpha=centroid_alpha, zorder=6)
                
                # Add markers at each feature point - larger markers
                ax.scatter(angles[:-1], centroid_center_loop[:-1], s=60,
                        color=dark_color, edgecolor='white', linewidth=1.5,
                        zorder=9)
                
                # Add legend entry for this centroid
                centroid_handle = plt.Line2D(
                    [0], [0], color=dark_color, lw=3, 
                    marker='o', markersize=8, markeredgecolor='white',
                    markeredgewidth=1.5, label=f'Centroid {lab+1}'
                )
                centroid_legend_handles.append(centroid_handle)
        
        # Set axis limits for fixed plotting scale
        ax.set_ylim(r_min, r_max)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add all legend handles - improved legend
        all_handles = legend_handles + centroid_legend_handles
        if all_handles and n_clusters > 1:
            ax.legend(handles=all_handles, loc='upper right', 
                     title="Clusters", title_fontsize=12,
                     frameon=True, framealpha=0.8, fontsize=10,
                     edgecolor='lightgray')
        
        # Add title as a separate text element at the top of the figure - no background
        if title:
            fig.suptitle(title, y=0.98, fontsize=18, fontweight='bold')
        
        return fig, ax
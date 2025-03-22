import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.colors import to_rgba

class IntervalRadarChart:
    """Class for radar chart visualization of interval data"""

    @staticmethod
    def visualize(intervals=None, centroids=None, labels=None, max_samples_per_cluster=None,
                  figsize=(10, 10), title="Interval Radar Chart", 
                  feature_names=None, alpha=0.2, centroid_alpha=0.4, margin=0.1):
        """
        Unified visualization function for radar chart with interval data
        
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
        ax = fig.add_subplot(111, polar=True)
            
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
            feature_names = [f"Feature {i+1}" for i in range(n_features)]
        elif len(feature_names) < n_features:
            # Extend feature names if fewer than n_features are provided
            feature_names = list(feature_names) + [f"Feature {i+1}" for i in range(len(feature_names), n_features)]
        
        # Setup the angles for the radar chart (equally spaced)
        angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
        # Close the loop for plotting
        angles.append(angles[0])
        
        # Extract data - using original values without normalization
        if intervals is not None:
            data_lower = intervals[:, :, 0]
            data_upper = intervals[:, :, 1]
            data_center = (data_lower + data_upper) / 2
            
            # Find min/max values for reference
            all_mins = np.min(data_lower, axis=0)
            all_maxs = np.max(data_upper, axis=0)
        else:
            all_mins = np.min(centroids[:, :, 0], axis=0)
            all_maxs = np.max(centroids[:, :, 1], axis=0)
            
        # Update min/max with centroids if available
        if centroids is not None and intervals is not None:
            centroid_mins = np.min(centroids[:, :, 0], axis=0)
            centroid_maxs = np.max(centroids[:, :, 1], axis=0)
            all_mins = np.minimum(all_mins, centroid_mins)
            all_maxs = np.maximum(all_maxs, centroid_maxs)
        
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
        
        # Find the global min and max across all features
        global_min = np.min(all_mins)
        global_max = np.max(all_maxs)
        
        # Calculate the radius range to accommodate negative values
        radius_range = global_max - global_min
        # Add a buffer to both ends
        buffer = radius_range * margin
        r_min = global_min - buffer
        r_max = global_max + buffer
        
        # Set radius offset to handle negative values
        # This shifts all values to ensure they're positive for plotting
        radius_offset = 0
        if global_min < 0:
            radius_offset = -global_min + buffer
        
        # Set up grid - always drawn now
        # Calculate levels for grid lines
        grid_levels = 5
        
        # Create grid levels that include both negative and positive values
        if global_min < 0 and global_max > 0:
            # Calculate reasonable grid steps
            total_range = r_max - r_min
            grid_step = total_range / grid_levels
            
            # Create grid values centered around zero if possible
            neg_levels = int(np.ceil(abs(r_min) / grid_step))
            pos_levels = int(np.ceil(r_max / grid_step))
            
            # Generate grid values
            grid_values = []
            for i in range(-neg_levels, pos_levels + 1):
                val = i * grid_step
                if r_min <= val <= r_max:
                    grid_values.append(val)
            
            # Convert to radii for plotting (shift by offset)
            grid_radii = [val + radius_offset for val in grid_values]
        else:
            # If all values are negative or all positive, use regular spacing
            grid_radii = np.linspace(r_min + radius_offset, r_max + radius_offset, grid_levels + 1)[1:]
            grid_values = np.linspace(r_min, r_max, grid_levels + 1)[1:]
        
        # Draw circular grid lines - DARKER GRID LINES
        for i, radius in enumerate(grid_radii):
            # Draw a complete circle for each grid level - increased alpha and linewidth
            circle = plt.Circle((0, 0), radius, transform=ax.transData._b, 
                             fill=False, edgecolor='gray', alpha=0.3,  # Increased alpha from 0.15 to 0.3
                             linestyle='-', linewidth=1.0)  # Increased linewidth from 0.7 to 1.0
            ax.add_patch(circle)
            
            # Add labels with the actual value (not the shifted radius)
            label_value = grid_values[i]
            
            # Format label based on magnitude
            if abs(label_value) >= 1000:
                label = f"{label_value/1000:.0f}k"
            elif abs(label_value) >= 100:
                label = f"{label_value:.0f}"
            elif abs(label_value) >= 10:
                label = f"{label_value:.1f}"
            else:
                label = f"{label_value:.2f}"
            
            # Darker labels with stronger background
            ax.text(angles[0], radius, label, 
                 color='black', fontsize=9, ha='left', va='bottom',  # Changed color to black and increased font size
                 alpha=1.0, fontweight='bold',  # Full opacity
                 bbox=dict(facecolor='white', alpha=0.8, pad=1, edgecolor='lightgray'))  # Added edge color
        
        # Draw radial lines for each feature - DARKER LINES
        for i in range(n_features):
            # Draw radial lines from min radius to max radius - increased alpha and linewidth
            ax.plot([angles[i], angles[i]], [r_min + radius_offset, r_max + radius_offset], 
                 color='gray', alpha=0.5, linestyle='-', linewidth=1.0)  # Increased alpha from 0.25 to 0.5
            
            # Improved label positioning with darker text - always showing labels now
            angle_rad = angles[i]
            # Set label distance slightly beyond the maximum radius
            label_distance = r_max + radius_offset + buffer * 0.5
            
            # Calculate text alignment based on angle
            if 0 <= angle_rad < np.pi/4 or 7*np.pi/4 <= angle_rad <= 2*np.pi:
                ha, va = 'left', 'center'
            elif np.pi/4 <= angle_rad < 3*np.pi/4:
                ha, va = 'center', 'bottom'
            elif 3*np.pi/4 <= angle_rad < 5*np.pi/4:
                ha, va = 'right', 'center'
            else:
                ha, va = 'center', 'top'
            
            # Place label at the angle with enhanced visibility
            ax.text(angle_rad, label_distance, feature_names[i], 
                 color='black', fontsize=11, fontweight='bold',  # Increased font size
                 horizontalalignment=ha, verticalalignment=va,
                 bbox=dict(facecolor='white', alpha=0.6, pad=2, edgecolor='none'))  # Added background
        
        # Draw data samples with offset for negative values
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
                    # Apply offset to handle negative values
                    offsetted_lower = data_lower[j] + radius_offset
                    offsetted_upper = data_upper[j] + radius_offset
                    offsetted_center = data_center[j] + radius_offset
                    
                    # Close the loop for each sample
                    line_data_lower = np.append(offsetted_lower, offsetted_lower[0])
                    line_data_upper = np.append(offsetted_upper, offsetted_upper[0])
                    line_data_center = np.append(offsetted_center, offsetted_center[0])
                    
                    # Plot center line with higher opacity
                    ax.plot(angles, line_data_center, color=line_color, 
                          alpha=0.7, linewidth=1.2, zorder=5)  # Increased from 0.5 to 0.7 opacity
                    
                    # Plot lower and upper bounds with higher opacity
                    ax.plot(angles, line_data_lower, color=line_color, 
                          alpha=0.5, linewidth=0.8, linestyle='--', zorder=4)  # Increased from 0.3 to 0.5 opacity
                    ax.plot(angles, line_data_upper, color=line_color, 
                          alpha=0.5, linewidth=0.8, linestyle='--', zorder=4)  # Increased from 0.3 to 0.5 opacity
                    
                    # Fill the area between upper and lower bounds with higher opacity
                    ax.fill_between(angles, line_data_lower, line_data_upper, 
                                 color=line_color, alpha=alpha, zorder=3)  # Using full alpha value, not reduced
                
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
                
                # Apply offset to handle negative values
                offsetted_lower = centroid_lower[idx] + radius_offset
                offsetted_upper = centroid_upper[idx] + radius_offset
                offsetted_center = centroid_center[idx] + radius_offset
                
                # Get a more prominent color for centroids
                base_color = cluster_colors[idx % len(cluster_colors)]
                dark_color = tuple([c*0.7 for c in base_color[:3]] + [1.0])  # Darker, fully opaque
                
                # Close the loop for centroids
                centroid_lower_loop = np.append(offsetted_lower, offsetted_lower[0])
                centroid_upper_loop = np.append(offsetted_upper, offsetted_upper[0])
                centroid_center_loop = np.append(offsetted_center, offsetted_center[0])
                
                # Plot center line for centroid with high visibility - always thicker line now
                line_width = 3.0  # Fixed thickness
                ax.plot(angles, centroid_center_loop, color=dark_color, 
                      alpha=1.0, linewidth=line_width, zorder=8)
                
                # Plot lower and upper bounds with dashed lines - higher opacity
                ax.plot(angles, centroid_lower_loop, color=dark_color, 
                      alpha=0.9, linewidth=1.2, linestyle='--', zorder=7)  # Increased from 0.7 to 0.9 opacity
                ax.plot(angles, centroid_upper_loop, color=dark_color, 
                      alpha=0.9, linewidth=1.2, linestyle='--', zorder=7)  # Increased from 0.7 to 0.9 opacity
                
                # Fill the area between upper and lower bounds with higher opacity
                ax.fill_between(angles, centroid_lower_loop, centroid_upper_loop, 
                             color=dark_color, alpha=centroid_alpha, zorder=6)
                
                # Add markers at each feature point - larger markers
                ax.scatter(angles[:-1], centroid_center_loop[:-1], s=60,  # Increased from 40 to 60
                        color=dark_color, edgecolor='white', linewidth=1.5,  # Thicker white edge
                        zorder=9)
                
                # Add legend entry for this centroid
                centroid_handle = plt.Line2D(
                    [0], [0], color=dark_color, lw=3, 
                    marker='o', markersize=8, markeredgecolor='white',  # Larger marker
                    markeredgewidth=1.5, label=f'Centroid {lab+1}'  # Thicker marker edge
                )
                centroid_legend_handles.append(centroid_handle)
        
        # Set axis limits to show all data plus some margin
        ax.set_ylim(r_min + radius_offset, r_max + radius_offset)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add all legend handles - improved legend
        all_handles = legend_handles + centroid_legend_handles
        if all_handles and n_clusters > 1:
            ax.legend(handles=all_handles, loc='upper right', 
                     title="Clusters", title_fontsize=12,  # Larger title
                     frameon=True, framealpha=0.8, fontsize=10,  # Higher opacity and larger font
                     edgecolor='lightgray')  # Added border
        
        # Add title if provided
        if title:
            ax.set_title(title, pad=30, fontsize=16, fontweight='bold')
        
        return fig, ax
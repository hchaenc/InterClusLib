import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.colors import to_rgba

class IntervalRadarChart:

    @staticmethod
    def plot_interval_radar(data=None, feature_names=None, clusters=None, centroids=None, ax=None, 
                   use_color=True, uncertainty_alpha=0.3, draw_grid=True, draw_labels=True,
                   grid_color='gray', label_color='black', label_fontsize=10, 
                   sample_opacity=0.4, highlight_centroids=True, max_samples_per_cluster=10,
                   title=None, title_fontsize=16, title_pad=35, show_outer_circle=False):
        """
        Plot radar chart for interval data with straight lines using original data values.
        Supports negative values in interval bounds.
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, polar=True)
            
        # Ensure either data or centroids (or both) are provided
        if data is None and centroids is None:
            raise ValueError("Either data or centroids must be provided")
            
        # Determine the number of features
        if data is not None:
            n_samples, n_features = data.shape[0], data.shape[1]
        else:
            n_features = centroids.shape[1]
            n_samples = 0
        
        # Auto-generate feature names if needed
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(n_features)]
        
        # Setup the angles for the radar chart (equally spaced)
        angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
        # Close the loop for plotting
        angles.append(angles[0])
        
        # Extract data - using original values without normalization
        if data is not None:
            data_lower = data[:, :, 0]
            data_upper = data[:, :, 1]
            data_center = (data_lower + data_upper) / 2
            
            # Find min/max values for reference
            all_mins = np.min(data_lower, axis=0)
            all_maxs = np.max(data_upper, axis=0)
        else:
            all_mins = np.min(centroids[:, :, 0], axis=0)
            all_maxs = np.max(centroids[:, :, 1], axis=0)
            
        # Update min/max with centroids if available
        if centroids is not None and data is not None:
            centroid_mins = np.min(centroids[:, :, 0], axis=0)
            centroid_maxs = np.max(centroids[:, :, 1], axis=0)
            all_mins = np.minimum(all_mins, centroid_mins)
            all_maxs = np.maximum(all_maxs, centroid_maxs)
        
        # Set up cluster information
        if data is not None:
            if clusters is None:
                clusters = np.zeros(n_samples, dtype=int)
            n_clusters = len(np.unique(clusters))
        elif centroids is not None:
            n_clusters = centroids.shape[0]
            clusters = np.array([]) if clusters is None else clusters
            
        # Generate distinct colors for clusters
        colors = plt.cm.get_cmap('tab10', n_clusters)
        cluster_colors = [colors(i) for i in range(n_clusters)]
        
        # Process centroids if provided
        if centroids is not None:
            if len(centroids.shape) != 3 or centroids.shape[1] != n_features or centroids.shape[2] != 2:
                raise ValueError(f"centroids should have shape (n_clusters, {n_features}, 2)")
            
            centroid_lower = centroids[:, :, 0]
            centroid_upper = centroids[:, :, 1]
            centroid_center = (centroid_lower + centroid_upper) / 2
        
        # Remove the default circle border if requested
        ax.spines['polar'].set_visible(show_outer_circle)
        
        # Find the global min and max across all features
        global_min = np.min(all_mins)
        global_max = np.max(all_maxs)
        
        # Calculate the radius range to accommodate negative values
        radius_range = global_max - global_min
        # Add a small buffer (10%) to both ends
        buffer = radius_range * 0.1
        r_min = global_min - buffer
        r_max = global_max + buffer
        
        # Set radius offset to handle negative values
        # This is the key modification - we'll shift all values by this offset
        radius_offset = 0
        if global_min < 0:
            radius_offset = -global_min + buffer
        
        # Set up grid
        if draw_grid:
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
            
            # Draw circular grid lines
            for i, radius in enumerate(grid_radii):
                # Draw a complete circle for each grid level
                circle = plt.Circle((0, 0), radius, transform=ax.transData._b, 
                                 fill=False, edgecolor=grid_color, alpha=0.15,
                                 linestyle='-', linewidth=0.7)
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
                
                ax.text(angles[0], radius, label, 
                     color=grid_color, fontsize=8, ha='left', va='bottom',
                     alpha=0.9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.6, pad=1, edgecolor='none'))
            
            # Draw radial lines for each feature
            for i in range(n_features):
                # Draw radial lines from min radius to max radius
                ax.plot([angles[i], angles[i]], [r_min + radius_offset, r_max + radius_offset], 
                     color=grid_color, alpha=0.25, linestyle='-', linewidth=0.8)
                
                # Improved label positioning
                if draw_labels:
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
                         color=label_color, fontsize=label_fontsize, fontweight='bold',
                         horizontalalignment=ha, verticalalignment=va)
        
        # Draw data samples with offset for negative values
        if data is not None:
            # For each cluster, select a limited number of samples to display
            for cluster_id in range(n_clusters):
                cluster_indices = np.where(clusters == cluster_id)[0]
                
                # Limit samples per cluster to reduce visual clutter
                if len(cluster_indices) > max_samples_per_cluster:
                    np.random.seed(42)  # For reproducibility
                    cluster_indices = np.random.choice(cluster_indices, max_samples_per_cluster, replace=False)
                
                for j in cluster_indices:
                    # Apply offset to handle negative values
                    offsetted_lower = data_lower[j] + radius_offset
                    offsetted_upper = data_upper[j] + radius_offset
                    offsetted_center = data_center[j] + radius_offset
                    
                    # Close the loop for each sample
                    line_data_lower = np.append(offsetted_lower, offsetted_lower[0])
                    line_data_upper = np.append(offsetted_upper, offsetted_upper[0])
                    line_data_center = np.append(offsetted_center, offsetted_center[0])
                    
                    # Get color for this cluster
                    line_color = cluster_colors[cluster_id]
                    
                    # Plot center line with reduced opacity
                    ax.plot(angles, line_data_center, color=line_color, 
                          alpha=sample_opacity, linewidth=1.0, zorder=5)
                    
                    # Plot lower and upper bounds with lower opacity
                    ax.plot(angles, line_data_lower, color=line_color, 
                          alpha=sample_opacity*0.6, linewidth=0.5, linestyle='--', zorder=4)
                    ax.plot(angles, line_data_upper, color=line_color, 
                          alpha=sample_opacity*0.6, linewidth=0.5, linestyle='--', zorder=4)
                    
                    # Fill the area between upper and lower bounds
                    ax.fill_between(angles, line_data_lower, line_data_upper, 
                                 color=line_color, alpha=uncertainty_alpha*0.4, zorder=3)
        
        # Draw centroids with enhanced visibility
        if centroids is not None:
            for i in range(n_clusters):
                # Skip if no data points in this cluster
                if data is not None and np.sum(clusters == i) == 0:
                    continue
                
                # Apply offset to handle negative values
                offsetted_lower = centroid_lower[i] + radius_offset
                offsetted_upper = centroid_upper[i] + radius_offset
                offsetted_center = centroid_center[i] + radius_offset
                
                # Get a more prominent color for centroids
                base_color = cluster_colors[i]
                dark_color = tuple([c*0.7 for c in base_color[:3]] + [1.0])  # Darker, fully opaque
                
                # Close the loop for centroids
                centroid_lower_loop = np.append(offsetted_lower, offsetted_lower[0])
                centroid_upper_loop = np.append(offsetted_upper, offsetted_upper[0])
                centroid_center_loop = np.append(offsetted_center, offsetted_center[0])
                
                # Plot center line for centroid with high visibility
                line_width = 2.5 if highlight_centroids else 1.5
                ax.plot(angles, centroid_center_loop, color=dark_color, 
                      alpha=1.0, linewidth=line_width, zorder=8)
                
                # Plot lower and upper bounds with dashed lines
                ax.plot(angles, centroid_lower_loop, color=dark_color, 
                      alpha=0.7, linewidth=1.0, linestyle='--', zorder=7)
                ax.plot(angles, centroid_upper_loop, color=dark_color, 
                      alpha=0.7, linewidth=1.0, linestyle='--', zorder=7)
                
                # Fill the area between upper and lower bounds
                ax.fill_between(angles, centroid_lower_loop, centroid_upper_loop, 
                             color=dark_color, alpha=uncertainty_alpha*0.8, zorder=6)
                
                # Add markers at each feature point
                ax.scatter(angles[:-1], centroid_center_loop[:-1], s=40, 
                        color=dark_color, edgecolor='white', linewidth=1, 
                        zorder=9)
        
        # Set axis limits to show all data plus some margin
        ax.set_ylim(r_min + radius_offset, r_max + radius_offset)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a legend in the original position (upper right corner)
        if use_color and n_clusters > 1:
            handles = []
            for i in range(n_clusters):
                base_color = cluster_colors[i]
                
                # Create legend entries - standard cluster lines
                sample_line = plt.Line2D([0], [0], color=base_color, lw=1.5, 
                                     label=f'Cluster {i+1}')
                handles.append(sample_line)
            
            if handles:
                # Position legend in the original position (upper right)
                legend = ax.legend(handles=handles, loc='upper right', 
                               frameon=True, fontsize=9, framealpha=0.9)
                legend.set_zorder(100)
        
        # Add title with proper spacing to avoid feature label overlaps
        if title:
            ax.set_title(title, pad=title_pad, fontsize=title_fontsize, fontweight='bold')
        
        return ax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from scipy.cluster.hierarchy import fcluster
import matplotlib.patches as patches
from interClusLib.visualization.AbstractIntervalVisualization import AbstractIntervalVisualization

class Dendrogram(AbstractIntervalVisualization):
    """
    Class for hierarchical clustering dendrogram visualization.
    
    Provides methods to visualize the hierarchical structure of clusters
    produced by hierarchical clustering algorithms.
    """
    
    @classmethod
    def visualize(cls, dendrogram_data, figsize=(10, 6), title="Hierarchical Clustering Dendrogram", 
                  color_threshold=None, orientation='top', 
                  leaf_font_size=10, show_leaf_counts=True, truncate_mode=None,
                  p=30, leaf_rotation=90., count_sort=False,
                  show_distances=False, n_clusters=None):
        """
        Visualize the hierarchical clustering as a dendrogram.
        
        Parameters:
        -----------
        dendrogram_data : dict
            Dictionary containing dendrogram data from model.get_dendrogram_data():
            - 'linkage_matrix': The linkage matrix for scipy dendrogram
            - 'labels': Cluster labels
            - 'n_leaves': Number of leaves in the tree
            - 'children': merge history
            - 'distances': distances at which merges occurred
        figsize : tuple, default=(10, 6)
            Figure size (width, height) in inches
        title : str, default="Hierarchical Clustering Dendrogram"
            Title for the plot
        color_threshold : float or None, default=None
            Color threshold for the dendrogram coloring
            If None, no coloring is applied beyond what scipy.dendrogram provides
        orientation : str, default='top'
            The direction to plot the dendrogram: 'top', 'right', 'bottom', or 'left'
        leaf_font_size : int, default=10
            Font size for the leaf labels
        show_leaf_counts : bool, default=True
            When True, leaf nodes display sample counts for each cluster
        truncate_mode : str or None, default=None
            Truncation mode for the dendrogram: 'lastp', 'level', or None
        p : int or None, default=30
            P parameter for truncation (used with truncate_mode)
        leaf_rotation : float, default=90.
            Rotation angle for leaf labels
        count_sort : bool or str, default=False
            If True, sort by leaf count. If 'ascending' or 'descending',
            sort in specified order
        show_distances : bool, default=False
            Whether to display the distances at each branch
        n_clusters : int or None, default=None
            Number of clusters to create distinct colors for.
            If provided, automatically calculates color_threshold.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The matplotlib figure object
        ax : matplotlib.axes.Axes
            The matplotlib axes object
        dendrogram_output : dict
            Dictionary containing the computed dendrogram output from scipy
        """
        # Validate input
        required_keys = ['linkage_matrix', 'n_leaves']
        for key in required_keys:
            if key not in dendrogram_data:
                raise ValueError(f"dendrogram_data missing required key: {key}")
        
        # Extract data
        linkage_matrix = dendrogram_data['linkage_matrix']
        n_leaves = dendrogram_data['n_leaves']
        
        # Calculate color threshold to get exact number of clusters if requested
        if n_clusters is not None and color_threshold is None:
            if n_clusters > 1 and n_clusters <= n_leaves:
                # Use binary search to find the threshold that produces the exact cluster count
                max_dist = np.max(linkage_matrix[:, 2])
                min_dist = 0
                
                # Initialize a reasonable threshold
                current_threshold = max_dist / 2
                current_clusters = len(np.unique(
                    fcluster(linkage_matrix, current_threshold, criterion='distance')
                ))
                
                # Binary search for appropriate threshold
                iterations = 0
                max_iterations = 50  # Prevent infinite loops
                
                while current_clusters != n_clusters and iterations < max_iterations:
                    iterations += 1
                    
                    if current_clusters < n_clusters:
                        # Threshold too high, need to lower
                        max_dist = current_threshold
                    else:
                        # Threshold too low, need to raise
                        min_dist = current_threshold
                    
                    # Update threshold
                    current_threshold = (min_dist + max_dist) / 2
                    current_clusters = len(np.unique(
                        fcluster(linkage_matrix, current_threshold, criterion='distance')
                    ))
                
                # Fine-tune threshold to ensure correct cluster count
                while True:
                    test_threshold = current_threshold * 0.9999  # Slightly lower threshold
                    test_clusters = len(np.unique(
                        fcluster(linkage_matrix, test_threshold, criterion='distance')
                    ))
                    
                    if test_clusters > n_clusters:
                        # Found appropriate threshold
                        break
                    else:
                        current_threshold = test_threshold
                
                color_threshold = current_threshold
            else:
                # Handle edge cases
                if n_clusters <= 1:
                    # All nodes as one cluster
                    color_threshold = np.max(linkage_matrix[:, 2]) * 1.1
                else:
                    # Each leaf as a separate cluster
                    color_threshold = 0
        
        # Set up the figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the dendrogram
        dendrogram_output = scipy_dendrogram(
            Z=linkage_matrix,
            p=p,
            ax=ax,
            orientation=orientation,
            leaf_font_size=leaf_font_size,
            show_leaf_counts=show_leaf_counts,
            truncate_mode=truncate_mode,
            leaf_rotation=leaf_rotation,
            count_sort=count_sort,
            color_threshold=color_threshold,
            above_threshold_color='black'
        )
        
        # Validate actual cluster count
        if n_clusters is not None:
            actual_clusters = len(np.unique(
                fcluster(linkage_matrix, color_threshold, criterion='distance')
            ))
            print(f"Requested clusters: {n_clusters}, Actual clusters: {actual_clusters}")
            title = f"{title} (requested n_clusters={n_clusters}, actual={actual_clusters})"
        
        # Determine axis orientation
        if orientation in ['top', 'bottom']:
            distance_axis = ax.yaxis
            sample_axis = ax.xaxis
        else:  # orientation in ['left', 'right']
            distance_axis = ax.xaxis
            sample_axis = ax.yaxis
        
        # Customize axis labels
        if orientation in ['top', 'bottom']:
            ax.set_xlabel('Samples')
            ax.set_ylabel('Distance')
        else:
            ax.set_xlabel('Distance')
            ax.set_ylabel('Samples')
        
        # Add title
        ax.set_title(title)
        
        # Format distance ticks if needed
        if show_distances:
            # Add more distance ticks for better readability
            distance_axis.set_major_locator(plt.MaxNLocator(10))
            plt.grid(axis='y' if orientation in ['top', 'bottom'] else 'x', linestyle='--', alpha=0.7)
        
        # Adjust layout and return figure, axes and output data
        plt.tight_layout()
        
        return fig, ax, dendrogram_output
    
    @classmethod
    def visualize_intervals(cls, intervals=None, centroids=None, labels=None, max_samples_per_cluster=None,
                 figsize=(8, 6), title="Cluster Intervals", feature_names=None, 
                 alpha=0.7, centroid_alpha=0.9, margin=0.1, orientation='horizontal',
                 cluster_order=None, sort_by='feature', colors=None, **kwargs):
        """
        Visualize interval data for each cluster as a separate subplot.
        Implements the abstract visualize method from IntervalVisualization.
        
        Parameters:
        :param intervals: Interval data with shape (n_samples, n_dims, 2), required
                         The last dimension represents [lower, upper]
        :param centroids: Centroid data with shape (n_clusters, n_dims, 2), can be None
                         The last dimension represents [lower, upper]
        :param labels: Cluster labels for each sample with shape (n_samples,), required
        :param max_samples_per_cluster: Maximum number of samples to display per cluster, default is None (show all samples)
        :param figsize: Figure size as tuple (width, height), default (8, 6)
        :param title: Main title for the plot, default "Cluster Intervals"
        :param feature_names: Names for each feature dimension, default is None, will auto-generate names
        :param alpha: Transparency for interval bars, default 0.7
        :param centroid_alpha: Transparency for centroids, default 0.9
        :param margin: Margin around axis limits, default 0.1
        :param orientation: 'horizontal' or 'vertical' orientation for interval bars, default 'horizontal'
        :param cluster_order: Custom order for plotting clusters, default None (ascending order)
        :param sort_by: How to sort intervals: 'feature', 'lower', 'upper', 'width', default 'feature'
        :param colors: List of colors for each cluster, default None (uses viridis colormap)
        :param kwargs: Additional parameters specific to this visualization
        :return: fig, axes - matplotlib figure and axes objects
        """
        # Validate required inputs
        if intervals is None:
            raise ValueError("The 'intervals' parameter is required for this visualization")
            
        if labels is None:
            raise ValueError("The 'labels' parameter is required for this visualization")
            
        # Validate data formats
        cls.validate_intervals(intervals)
        if centroids is not None:
            cls.validate_centroids(centroids)
            
        # Determine number of clusters and features
        unique_clusters = np.unique(labels)
        n_clusters = len(unique_clusters)
        n_features = intervals.shape[1]
        
        # Generate feature names if not provided
        feature_names = cls.generate_feature_names(n_features, feature_names, prefix="Feature ")
        
        # Set up cluster order
        if cluster_order is None:
            cluster_order = sorted(unique_clusters)
        else:
            # Validate cluster_order
            if set(cluster_order) != set(unique_clusters):
                raise ValueError("cluster_order must contain all unique cluster labels")
        
        # Set up colors
        if colors is None:
            colors = cls.generate_cluster_colors(n_clusters, 'viridis')
        elif len(colors) < n_clusters:
            # Extend colors if needed
            additional_colors = cls.generate_cluster_colors(n_clusters - len(colors), 'viridis')
            colors = list(colors) + additional_colors
        
        # Calculate appropriate figure layout
        n_rows = int(np.ceil(np.sqrt(n_clusters)))
        n_cols = int(np.ceil(n_clusters / n_rows))
        
        # Create figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                                squeeze=False, sharex=True, sharey=True)
        axes = axes.flatten()
        
        # Hide unused subplots
        for i in range(n_clusters, len(axes)):
            axes[i].set_visible(False)
        
        # Plot intervals for each cluster
        for i, cluster_idx in enumerate(cluster_order):
            ax = axes[i]
            
            # Get intervals for this cluster
            cluster_mask = (labels == cluster_idx)
            cluster_intervals = intervals[cluster_mask]
            
            # Plot each feature dimension
            for j in range(n_features):
                feature_intervals = cluster_intervals[:, j, :]
                
                # Determine sorting method
                if sort_by == 'lower':
                    sort_idx = np.argsort(feature_intervals[:, 0])
                elif sort_by == 'upper':
                    sort_idx = np.argsort(feature_intervals[:, 1])
                elif sort_by == 'width':
                    widths = feature_intervals[:, 1] - feature_intervals[:, 0]
                    sort_idx = np.argsort(widths)
                else:  # Default to 'feature'
                    sort_idx = np.arange(len(feature_intervals))
                
                # Sort intervals
                feature_intervals = feature_intervals[sort_idx]
                
                # Create interval position markers
                n_intervals = len(feature_intervals)
                positions = np.arange(n_intervals)
                
                # Adjust position based on feature index
                if orientation == 'horizontal':
                    # Offset different features for clarity
                    y_positions = positions + (j * 0.1)
                    
                    # Plot horizontal intervals
                    for k, (lower, upper) in enumerate(feature_intervals):
                        ax.plot([lower, upper], [y_positions[k], y_positions[k]], 
                               color=colors[i], linewidth=2, alpha=alpha)
                        ax.plot([lower, lower], [y_positions[k]-0.05, y_positions[k]+0.05], 
                               color=colors[i], linewidth=2, alpha=alpha)
                        ax.plot([upper, upper], [y_positions[k]-0.05, y_positions[k]+0.05], 
                               color=colors[i], linewidth=2, alpha=alpha)
                
                else:  # vertical orientation
                    # Offset different features for clarity
                    x_positions = positions + (j * 0.1)
                    
                    # Plot vertical intervals
                    for k, (lower, upper) in enumerate(feature_intervals):
                        ax.plot([x_positions[k], x_positions[k]], [lower, upper], 
                               color=colors[i], linewidth=2, alpha=alpha)
                        ax.plot([x_positions[k]-0.05, x_positions[k]+0.05], [lower, lower], 
                               color=colors[i], linewidth=2, alpha=alpha)
                        ax.plot([x_positions[k]-0.05, x_positions[k]+0.05], [upper, upper], 
                               color=colors[i], linewidth=2, alpha=alpha)
            
            # Plot centroid if available
            if centroids is not None:
                if i < len(centroids):  # Safety check
                    centroid = centroids[i]
                    
                    for j in range(n_features):
                        c_lower, c_upper = centroid[j]
                        
                        if orientation == 'horizontal':
                            # Plot horizontal centroid line
                            y_pos = -1  # Position below data points
                            ax.plot([c_lower, c_upper], [y_pos, y_pos], 
                                   color='red', linewidth=3, alpha=centroid_alpha)
                            ax.plot([c_lower, c_lower], [y_pos-0.2, y_pos+0.2], 
                                   color='red', linewidth=3, alpha=centroid_alpha)
                            ax.plot([c_upper, c_upper], [y_pos-0.2, y_pos+0.2], 
                                   color='red', linewidth=3, alpha=centroid_alpha)
                            
                            # Label
                            ax.text((c_lower + c_upper)/2, y_pos-0.5, 
                                   f'{feature_names[j]} centroid', 
                                   ha='center', va='top', color='red',
                                   fontweight='bold', fontsize=8)
                        
                        else:  # vertical orientation
                            # Plot vertical centroid line
                            x_pos = -1  # Position to the left of data points
                            ax.plot([x_pos, x_pos], [c_lower, c_upper], 
                                   color='red', linewidth=3, alpha=centroid_alpha)
                            ax.plot([x_pos-0.2, x_pos+0.2], [c_lower, c_lower], 
                                   color='red', linewidth=3, alpha=centroid_alpha)
                            ax.plot([x_pos-0.2, x_pos+0.2], [c_upper, c_upper], 
                                   color='red', linewidth=3, alpha=centroid_alpha)
                            
                            # Label
                            ax.text(x_pos-0.5, (c_lower + c_upper)/2, 
                                   f'{feature_names[j]} centroid', 
                                   ha='right', va='center', color='red',
                                   fontweight='bold', fontsize=8, rotation=90)
            
            # Set axis labels and title
            if orientation == 'horizontal':
                ax.set_xlabel('Value')
                ax.set_ylabel('Sample Index')
            else:
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Value')
            
            ax.set_title(f'Cluster {cluster_idx+1} (n={np.sum(cluster_mask)})')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Create a common legend for features
        custom_lines = [plt.Line2D([0], [0], color='black', lw=2, label=name) 
                       for name in feature_names]
        
        fig.legend(handles=custom_lines, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.05), ncol=min(5, n_features))
        
        # Set overall title
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust for the legend at bottom
        
        return fig, axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from interClusLib.visualization.IntervalVisualization import IntervalVisualization

class Interval2d(IntervalVisualization):
    """Implementation class for 2D interval visualization"""
    
    @classmethod
    def visualize(cls, intervals=None, centroids=None, labels=None, max_samples_per_cluster=None,
                  figsize=(8, 8), title="2D Intervals", alpha=0.3, centroid_alpha=0.6, 
                  margin=0.5, feature_names=None, fill_intervals=False):
        """
        Unified visualization function for 2D intervals with optional centroids
        
        Parameters:
        :param intervals: Interval data with shape (n_samples, n_dims, 2), can be None if only plotting centroids
                         - If n_dims = 1: Drawn as squares
                         - If n_dims = 2: Used as is
                         - If n_dims > 2: Only the first 2 dimensions will be used
        :param centroids: Centroid data with shape (n_clusters, n_dims, 2), can be None if only plotting intervals
                         Dimensions will be processed the same as intervals
        :param labels: Optional, array of shape (n_samples,) for cluster labels or categories
        :param figsize: Figure size, default is (8, 8)
        :param title: Figure title
        :param alpha: Transparency for regular intervals, default is 0.3
        :param centroid_alpha: Transparency for centroids, default is 0.6
        :param margin: Margin around axis limits, default is 0.5
        :param feature_names: List of feature names, default is None, will auto-generate ["x1", "x2"]
        :param fill_intervals: Whether to fill the intervals with color, default is False
        :param max_samples_per_cluster: Maximum number of samples to display per cluster, default is None (show all samples)
        :return: fig, ax - matplotlib figure and axes objects
        """
        # Validate input: at least one of intervals or centroids must be provided
        if intervals is None and centroids is None:
            raise ValueError("At least one of 'intervals' or 'centroids' must be provided")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Process intervals if provided
        interval_legend_handles = []
        if intervals is not None:
            processed_intervals = cls._process_intervals(intervals)
            interval_legend_handles = cls._draw_2d_intervals(ax, processed_intervals, labels, 
                                                          alpha=alpha, fill=fill_intervals,
                                                          max_samples_per_cluster=max_samples_per_cluster)
        else:
            processed_intervals = None
            
        # Process centroids if provided
        if centroids is not None:
            processed_centroids = cls._process_intervals(centroids)
            
            # Get cluster information
            if intervals is not None and labels is not None:
                _, unique_labels, n_clusters = cls.setup_cluster_info(intervals, labels, centroids)
            else:
                n_clusters = processed_centroids.shape[0]
                unique_labels = np.arange(n_clusters)
            
            # Generate colors for clusters
            cluster_colors = cls.generate_cluster_colors(n_clusters, 'viridis')
            
            centroid_legend_handles = []
            for i in range(n_clusters):
                # Get darker color for centroids
                base_color = np.array(cluster_colors[i])
                dark_color = np.clip(base_color * 0.7, 0, 1)
                dark_color[3] = 1.0  # Full opacity
                
                # Extract coordinates
                if processed_centroids.shape[1] == 1:  # 1D case
                    lower, upper = processed_centroids[i, 0, 0], processed_centroids[i, 0, 1]
                    x_lower, y_lower = lower, lower
                    width = height = upper - lower
                else:  # 2D case
                    x_lower, x_upper = processed_centroids[i, 0, 0], processed_centroids[i, 0, 1]
                    y_lower, y_upper = processed_centroids[i, 1, 0], processed_centroids[i, 1, 1]
                    width = x_upper - x_lower
                    height = y_upper - y_lower
                
                # Draw centroid rectangle
                rect = patches.Rectangle(
                    (x_lower, y_lower), width, height,
                    linewidth=3, edgecolor=dark_color, 
                    facecolor=dark_color if fill_intervals else 'none',
                    alpha=centroid_alpha if fill_intervals else 1.0,
                    fill=fill_intervals,  # Ensure filling and border are perfectly aligned
                    joinstyle='miter',  # Make corners sharper
                    capstyle='projecting',  # Ensure corners are fully filled
                    zorder=10
                )
                ax.add_patch(rect)
                
                # Create legend handle for this centroid
                label = f"Centroid {i+1}"
                
                # Modified to remove center point from legend representation
                centroid_handle = plt.Line2D(
                    [0], [0], color=dark_color,
                    linestyle='-', linewidth=3, label=label
                )
                centroid_legend_handles.append(centroid_handle)
            
            # Add all legend handles
            all_handles = interval_legend_handles + centroid_legend_handles
            if all_handles:
                ax.legend(handles=all_handles, loc="best", 
                         title="Clusters", title_fontsize=10,
                         frameon=True, framealpha=0.7, fontsize=9)
        # If we only have intervals (no centroids) but we have labels, add the legend directly
        elif intervals is not None and labels is not None and interval_legend_handles:
            ax.legend(handles=interval_legend_handles, loc="best", 
                     title="Clusters", title_fontsize=10,
                     frameon=True, framealpha=0.7, fontsize=9)
        
        # Determine axis limits based on available data
        if processed_intervals is not None:
            if processed_intervals.shape[1] == 1:  # 1D case
                lower_upper = processed_intervals[:, 0, :].ravel()
                xs = ys = lower_upper
            else:  # 2D case
                xs = processed_intervals[:, 0, :].ravel()  # x_lower, x_upper
                ys = processed_intervals[:, 1, :].ravel()  # y_lower, y_upper
        else:  # Only have centroids
            if processed_centroids.shape[1] == 1:  # 1D case
                lower_upper = processed_centroids[:, 0, :].ravel()
                xs = ys = lower_upper
            else:  # 2D case
                xs = processed_centroids[:, 0, :].ravel()
                ys = processed_centroids[:, 1, :].ravel()
            
        # Calculate axis limits based on the available data
        if processed_intervals is not None and centroids is not None:
            # If we have both, take the min/max across both datasets
            if processed_centroids.shape[1] == 1:  # 1D case
                cent_lower_upper = processed_centroids[:, 0, :].ravel()
                xs_centroids = ys_centroids = cent_lower_upper
            else:  # 2D case
                xs_centroids = processed_centroids[:, 0, :].ravel()
                ys_centroids = processed_centroids[:, 1, :].ravel()
            
            x_min = min(xs.min(), xs_centroids.min()) - margin
            x_max = max(xs.max(), xs_centroids.max()) + margin
            y_min = min(ys.min(), ys_centroids.min()) - margin
            y_max = max(ys.max(), ys_centroids.max()) + margin
        else:
            # Only use the data we have
            x_min, x_max = xs.min() - margin, xs.max() + margin
            y_min, y_max = ys.min() - margin, ys.max() + margin

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Set axis labels
        if feature_names is None:
            if processed_intervals is not None and processed_intervals.shape[1] == 1:
                feature_names = ["value", "value"]  # 1D case
            else:
                feature_names = ["x1", "x2"]
        elif len(feature_names) < 2:
            feature_names = list(feature_names) + [f"x{i+1}" for i in range(len(feature_names), 2)]
        
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title(title)
        
        # Set aspect ratio to equal for square visualization
        if processed_intervals is not None and processed_intervals.shape[1] == 1:
            ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def _process_intervals(intervals):
        """
        Process intervals to ensure they have the correct shape (n_samples, n_dims, 2)
        - For higher dimensions, will use only the first 2 dimensions
        
        Parameters:
        :param intervals: Input intervals with shape (n_samples, n_dims, 2)
        :return: Processed intervals with shape (n_samples, n_dims, 2) where n_dims is 1 or 2
        """
        # Only handle 3D array case (n_samples, n_dims, 2)
        if intervals.ndim != 3 or intervals.shape[2] != 2:
            raise ValueError(
                f"Interval data should have shape (n_samples, n_dims, 2). "
                f"Got {intervals.shape} instead."
            )
            
        n_samples, n_dims, width = intervals.shape
            
        if n_dims == 1 or n_dims == 2:
            # Already in correct format
            return intervals
        elif n_dims > 2:
            # Take only the first 2 dimensions
            return intervals[:, :2, :]
        else:
            raise ValueError(
                f"Input intervals must have at least 1 dimension. Got {n_dims}."
            )
    
    @staticmethod
    def _draw_2d_intervals(ax, intervals, labels=None, alpha=0.3, line_width=2, 
                           fill=False, max_samples_per_cluster=None):
        """
        Draw each 2D interval as a rectangle (or square for 1D) in the given axes 'ax'.
        
        Parameters:
        :param ax: Matplotlib axes object
        :param intervals: shape (n_samples, n_dims, 2) where n_dims is 1 or 2
        :param labels: Optional, shape (n_samples,) - cluster or category labels
        :param alpha: Transparency, default is 0.3
        :param line_width: Line width, default is 2
        :param fill: Whether to fill the intervals with color, default is False
        :param max_samples_per_cluster: Maximum number of samples to display per cluster, default is None (show all samples)
        :return: List of legend handles
        """
        n_dims = intervals.shape[1]
        legend_handles = []
        
        if labels is not None:
            unique_labels = np.unique(labels)
            cmap = plt.cm.get_cmap('viridis', len(unique_labels))

            for idx, lab in enumerate(unique_labels):
                mask = (labels == lab)
                sub_intervals_indices = np.where(mask)[0]
                
                # Limit samples per cluster if specified
                if max_samples_per_cluster is not None and len(sub_intervals_indices) > max_samples_per_cluster:
                    np.random.seed(42)  # For reproducibility
                    selected_indices = np.random.choice(sub_intervals_indices, max_samples_per_cluster, replace=False)
                    mask = np.zeros_like(mask)
                    mask[selected_indices] = True
                
                sub_intervals = intervals[mask]
                color = cmap(idx)
                
                for interval in sub_intervals:
                    if n_dims == 1:  # 1D intervals -> squares
                        lower, upper = interval[0, 0], interval[0, 1]
                        width = height = upper - lower
                        rect = patches.Rectangle(
                            (lower, lower), width, height,
                            linewidth=line_width+1, 
                            edgecolor=color, 
                            facecolor=color if fill else 'none',
                            alpha=alpha if fill else 1.0,
                            fill=fill,  # Ensure filling and border are perfectly aligned
                            joinstyle='miter',  # Make corners sharper
                            capstyle='projecting'  # Ensure corners are fully filled
                        )
                    else:  # 2D intervals -> rectangles
                        x_lower, x_upper = interval[0, 0], interval[0, 1]
                        y_lower, y_upper = interval[1, 0], interval[1, 1]
                        width = x_upper - x_lower
                        height = y_upper - y_lower
                        rect = patches.Rectangle(
                            (x_lower, y_lower), width, height,
                            linewidth=line_width+1, 
                            edgecolor=color, 
                            facecolor=color if fill else 'none',
                            alpha=alpha if fill else 1.0,
                            fill=fill,  # Ensure filling and border are perfectly aligned
                            joinstyle='miter',  # Make corners sharper
                            capstyle='projecting'  # Ensure corners are fully filled
                        )
                    ax.add_patch(rect)
                
                # Create legend handle for this cluster
                label = f"Cluster {lab+1}"
                
                # Create a custom legend handle with a square icon
                legend_handle = plt.Line2D(
                    [0], [0], color=color, lw=4, marker='s', 
                    markersize=12, markerfacecolor=color if fill else 'white', 
                    markeredgecolor=color, markeredgewidth=2,
                    label=label
                )
                legend_handles.append(legend_handle)

        else:
            # No labels, all intervals use the same color
            color = "blue"
            
            # If max_samples_per_cluster is specified, limit the number of intervals displayed
            if max_samples_per_cluster is not None and len(intervals) > max_samples_per_cluster:
                np.random.seed(42)  # For reproducibility
                selected_indices = np.random.choice(len(intervals), max_samples_per_cluster, replace=False)
                intervals_to_draw = intervals[selected_indices]
            else:
                intervals_to_draw = intervals
                
            for interval in intervals_to_draw:
                if n_dims == 1:  # 1D intervals -> squares
                    lower, upper = interval[0, 0], interval[0, 1]
                    width = height = upper - lower
                    rect = patches.Rectangle(
                        (lower, lower), width, height,
                        linewidth=line_width+1, 
                        edgecolor=color, 
                        facecolor=color if fill else 'none',
                        alpha=alpha if fill else 1.0,
                        fill=fill,  # Ensure filling and border are perfectly aligned
                        joinstyle='miter',  # Make corners sharper
                        capstyle='projecting'  # Ensure corners are fully filled
                    )
                else:  # 2D intervals -> rectangles
                    x_lower, x_upper = interval[0, 0], interval[0, 1]
                    y_lower, y_upper = interval[1, 0], interval[1, 1]
                    width = x_upper - x_lower
                    height = y_upper - y_lower
                    rect = patches.Rectangle(
                        (x_lower, y_lower), width, height,
                        linewidth=line_width+1, 
                        edgecolor=color, 
                        facecolor=color if fill else 'none',
                        alpha=alpha if fill else 1.0,
                        fill=fill,  # Ensure filling and border are perfectly aligned
                        joinstyle='miter',  # Make corners sharper
                        capstyle='projecting'  # Ensure corners are fully filled
                    )
                ax.add_patch(rect)
                
            # Add a single legend handle for all intervals
            legend_handle = plt.Line2D(
                [0], [0], color=color, lw=4, marker='s',
                markersize=12, markerfacecolor=color if fill else 'white', 
                markeredgecolor=color, markeredgewidth=2,
                label="Intervals"
            )
            legend_handles.append(legend_handle)
            
        return legend_handles
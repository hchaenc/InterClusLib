import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

class Interval3d:
    """Class for 3D interval visualization"""
    
    @staticmethod
    def visualize(intervals=None, centroids=None, labels=None, figsize=(8, 8), title="3D Intervals", 
                  alpha=0.3, centroid_alpha=0.6, margin=5, feature_names=None):
        """
        Unified visualization function for 3D intervals with optional centroids
        
        Parameters:
        :param intervals: Interval data with shape (n_samples, n_dims, 2), can be None if only plotting centroids
                         - If n_dims = 3: Used as is
                         - If n_dims > 3: Only the first 3 dimensions will be used
                         - If n_dims < 3: Will raise ValueError
        :param centroids: Centroid data with shape (n_clusters, n_dims, 2), can be None if only plotting intervals
                         Dimensions will be processed the same as intervals
        :param labels: Optional, array of shape (n_samples,) for cluster labels or categories
        :param figsize: Figure size, default is (8, 8)
        :param title: Figure title
        :param alpha: Transparency for regular intervals, default is 0.3
        :param centroid_alpha: Transparency for centroids, default is 0.6
        :param margin: Margin around axis limits, default is 5
        :param feature_names: List of feature names, default is None, will auto-generate ["x1", "x2", "x3"]
        :return: fig, ax - matplotlib figure and axes objects
        """
        # Validate input: at least one of intervals or centroids must be provided
        if intervals is None and centroids is None:
            raise ValueError("At least one of 'intervals' or 'centroids' must be provided")
            
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Process intervals if provided
        interval_legend_handles = []
        if intervals is not None:
            processed_intervals = Interval3d._process_intervals(intervals)
            interval_legend_handles = Interval3d.draw_3d_intervals(ax, processed_intervals, labels, alpha=alpha)
        else:
            processed_intervals = None
            
        # Process centroids if provided
        if centroids is not None:
            processed_centroids = Interval3d._process_intervals(centroids)
            
            # If we have both intervals and labels, ensure color consistency between intervals and centroids
            n_clusters = processed_centroids.shape[0]
            if labels is not None and intervals is not None:
                unique_labels = np.unique(labels)
                cmap = plt.cm.get_cmap('viridis', max(len(unique_labels), n_clusters))
            else:
                cmap = plt.cm.get_cmap('viridis', n_clusters)
            
            centroid_legend_handles = []
            for i in range(n_clusters):
                # Get darker color for centroids
                base_color = np.array(cmap(i))
                dark_color = np.clip(base_color * 0.7, 0, 1)
                dark_color[3] = 1.0  # Full opacity
                
                x_lower, x_upper = processed_centroids[i, 0, 0], processed_centroids[i, 0, 1]
                y_lower, y_upper = processed_centroids[i, 1, 0], processed_centroids[i, 1, 1]
                z_lower, z_upper = processed_centroids[i, 2, 0], processed_centroids[i, 2, 1]
                
                faces = Interval3d.get_cube_faces(x_lower, x_upper, y_lower, y_upper, z_lower, z_upper)
                cube = Poly3DCollection(
                    faces, 
                    edgecolor="black", 
                    facecolor=dark_color, 
                    linewidths=2, 
                    alpha=centroid_alpha, 
                    zsort='min'
                )
                ax.add_collection3d(cube)
                
                # Add marker at centroid center
                center_x = (x_lower + x_upper) / 2
                center_y = (y_lower + y_upper) / 2
                center_z = (z_lower + z_upper) / 2
                ax.scatter([center_x], [center_y], [center_z], 
                          color=dark_color, s=100, edgecolor='white', linewidth=1, zorder=10)
                
                # Create legend handle for this centroid
                label = f"Centroid {i+1}"
                
                centroid_handle = plt.Line2D(
                    [0], [0], color=dark_color, marker='o', 
                    markersize=8, markeredgecolor='white', markeredgewidth=1,
                    linestyle='-', linewidth=2, label=label
                )
                centroid_legend_handles.append(centroid_handle)
            
            # Add all legend handles
            all_handles = interval_legend_handles + centroid_legend_handles
            if all_handles:
                ax.legend(handles=all_handles, loc="upper right", 
                         title="Clusters", title_fontsize=10,
                         frameon=True, framealpha=0.7, fontsize=9)
        # If we only have intervals (no centroids) but we have labels, add the legend directly
        elif intervals is not None and labels is not None and interval_legend_handles:
            ax.legend(handles=interval_legend_handles, loc="upper right", 
                     title="Clusters", title_fontsize=10,
                     frameon=True, framealpha=0.7, fontsize=9)
        
        # Determine axis limits based on available data
        if processed_intervals is not None:
            xs = processed_intervals[:, 0, :].ravel()  # x_lower, x_upper
            ys = processed_intervals[:, 1, :].ravel()  # y_lower, y_upper
            zs = processed_intervals[:, 2, :].ravel()  # z_lower, z_upper
        else:  # Only have centroids
            xs = processed_centroids[:, 0, :].ravel()
            ys = processed_centroids[:, 1, :].ravel()
            zs = processed_centroids[:, 2, :].ravel()
            
        # Calculate axis limits based on the available data
        if processed_intervals is not None and centroids is not None:
            # If we have both, take the min/max across both datasets
            xs_centroids = processed_centroids[:, 0, :].ravel()
            ys_centroids = processed_centroids[:, 1, :].ravel()
            zs_centroids = processed_centroids[:, 2, :].ravel()
            
            x_min = min(xs.min(), xs_centroids.min()) - margin
            x_max = max(xs.max(), xs_centroids.max()) + margin
            y_min = min(ys.min(), ys_centroids.min()) - margin
            y_max = max(ys.max(), ys_centroids.max()) + margin
            z_min = min(zs.min(), zs_centroids.min()) - margin
            z_max = max(zs.max(), zs_centroids.max()) + margin
        else:
            # Only use the data we have
            x_min, x_max = xs.min() - margin, xs.max() + margin
            y_min, y_max = ys.min() - margin, ys.max() + margin
            z_min, z_max = zs.min() - margin, zs.max() + margin

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        # Set axis labels
        if feature_names is None:
            feature_names = ["x1", "x2", "x3"]
        elif len(feature_names) < 3:
            feature_names = list(feature_names) + [f"x{i+1}" for i in range(len(feature_names), 3)]
        
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_zlabel(feature_names[2])
        ax.set_title(title)
        
        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def _process_intervals(intervals):
        """
        Process intervals to ensure they have the correct shape (n_samples, 3, 2)
        - Rejects dimensions less than 3 with ValueError
        - Handles dimensions greater than 3 by taking only the first 3
        
        Parameters:
        :param intervals: Input intervals with shape (n_samples, n_dims, 2)
        :return: Processed intervals with shape (n_samples, 3, 2)
        """
        # Only handle 3D array case (n_samples, n_dims, 2)
        if intervals.ndim == 3:
            n_samples, n_dims, width = intervals.shape
            
            if width != 2:
                raise ValueError(
                    f"Last dimension of intervals must be 2 (lower, upper). Got {width}."
                )
                
            if n_dims < 3:
                raise ValueError(
                    f"Input intervals must have at least 3 dimensions. Got {n_dims}."
                )
                
            elif n_dims > 3:
                # Take only the first 3 dimensions
                return intervals[:, :3, :]
                
            else:  # n_dims == 3
                return intervals
                
        else:
            raise ValueError(
                f"Unexpected number of dimensions in interval array: {intervals.ndim}. "
                "Expected 3 dimensions (n_samples, n_dims, 2)."
            )
    
    @staticmethod
    def get_cube_faces(x_lower, x_upper, y_lower, y_upper, z_lower, z_upper):
        """
        Given lower and upper bounds in x, y, z dimensions,
        return a list of 6 faces (polygons), each face is a list of 4 corner points (x,y,z).
        
        Parameters:
        :param x_lower, x_upper: Lower and upper bounds on x axis
        :param y_lower, y_upper: Lower and upper bounds on y axis
        :param z_lower, z_upper: Lower and upper bounds on z axis
        :return: List of six faces, each face is a list of four vertex coordinates
        """
        # 8 corners
        c000 = (x_lower, y_lower, z_lower)
        c001 = (x_lower, y_lower, z_upper)
        c010 = (x_lower, y_upper, z_lower)
        c011 = (x_lower, y_upper, z_upper)
        c100 = (x_upper, y_lower, z_lower)
        c101 = (x_upper, y_lower, z_upper)
        c110 = (x_upper, y_upper, z_lower)
        c111 = (x_upper, y_upper, z_upper)

        # 6 faces, each face is a list of 4 corners in either clockwise or counterclockwise order
        faces = [
            [c000, c010, c011, c001],  # x_lower face
            [c100, c101, c111, c110],  # x_upper face
            [c000, c001, c101, c100],  # y_lower face
            [c010, c011, c111, c110],  # y_upper face
            [c000, c100, c110, c010],  # z_lower face
            [c001, c101, c111, c011],  # z_upper face
        ]
        return faces

    @staticmethod
    def draw_3d_intervals(ax, intervals, labels=None, alpha=0.3, line_width=1):
        """
        Draw each 3D interval (x_lower, x_upper; y_lower, y_upper; z_lower, z_upper)
        as a cuboid in the given 3D axes 'ax'.
        
        Parameters:
        :param ax: 3D matplotlib axes object (e.g. from fig.add_subplot(projection='3d'))
        :param intervals: shape (n_samples, 3, 2)
        :param labels: Optional, shape (n_samples,) - cluster or category labels
        :param alpha: Transparency, default is 0.3
        :param line_width: Line width, default is 1
        :return: List of legend handles
        """
        # Check data structure
        if intervals.ndim != 3 or intervals.shape[1] != 3 or intervals.shape[2] != 2:
            raise ValueError(
                "Expected intervals to have shape (n_samples, 3, 2). "
                f"Got {intervals.shape} instead."
            )

        legend_handles = []
        if labels is not None:
            unique_labels = np.unique(labels)
            cmap = plt.cm.get_cmap('viridis', len(unique_labels))

            for idx, lab in enumerate(unique_labels):
                mask = (labels == lab)
                sub_intervals = intervals[mask]
                color = cmap(idx)

                for interval3d in sub_intervals:
                    x_lower, x_upper = interval3d[0, 0], interval3d[0, 1]
                    y_lower, y_upper = interval3d[1, 0], interval3d[1, 1]
                    z_lower, z_upper = interval3d[2, 0], interval3d[2, 1]

                    faces = Interval3d.get_cube_faces(x_lower, x_upper, y_lower, y_upper, z_lower, z_upper)
                    # Poly3DCollection expects a list of faces, each face is a list of (x,y,z) points
                    cube = Poly3DCollection(
                        faces, 
                        edgecolor="black", 
                        facecolor=color, 
                        linewidths=line_width, 
                        alpha=alpha, 
                        zsort='min'
                    )
                    ax.add_collection3d(cube)

                # Create legend handle for this cluster
                label = f"Cluster {lab+1}"
                
                # Create a custom legend handle with a cube-like icon
                legend_handle = plt.Line2D(
                    [0], [0], color=color, lw=3, marker='s', 
                    markersize=10, markerfacecolor=color, markeredgecolor='black',
                    label=label
                )
                legend_handles.append(legend_handle)

        else:
            color = "blue"
            for interval3d in intervals:
                x_lower, x_upper = interval3d[0, 0], interval3d[0, 1]
                y_lower, y_upper = interval3d[1, 0], interval3d[1, 1]
                z_lower, z_upper = interval3d[2, 0], interval3d[2, 1]

                faces = Interval3d.get_cube_faces(x_lower, x_upper, y_lower, y_upper, z_lower, z_upper)
                cube = Poly3DCollection(
                    faces, 
                    edgecolor="black", 
                    facecolor=color, 
                    linewidths=line_width, 
                    alpha=alpha, 
                    zsort='min'
                )
                ax.add_collection3d(cube)
                
            # Add a single legend handle for all intervals
            legend_handle = plt.Line2D(
                [0], [0], color=color, lw=3, marker='s',
                markersize=10, markerfacecolor=color, markeredgecolor='black',
                label="Intervals"
            )
            legend_handles.append(legend_handle)
            
        return legend_handles
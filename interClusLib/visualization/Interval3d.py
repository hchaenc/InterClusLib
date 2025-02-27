import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Interval3d:

    @staticmethod
    def get_cube_faces(x_lower, x_upper, y_lower, y_upper, z_lower, z_upper):
        """
        Given lower and upper bounds in x, y, z dimensions,
        return a list of 6 faces (polygons), each face is a list of 4 corner points (x,y,z).
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

    def draw_3d_interval_cubes(ax, intervals, labels=None):
        """
        Draw each 3D interval (x_lower, x_upper; y_lower, y_upper; z_lower, z_upper)
        as a cuboid in the given 3D axes 'ax'.

        :param ax: A 3D matplotlib Axes object (e.g. from fig.add_subplot(projection='3d'))
        :param intervals: shape (n_samples, 3, 2)
        :param labels: optional, shape (n_samples,) - cluster or category labels
        """
        # check data structure
        if intervals.ndim != 3 or intervals.shape[1] != 3 or intervals.shape[2] != 2:
            raise ValueError(
                "Expected intervals to have shape (n_samples, 3, 2). "
                f"Got {intervals.shape} instead."
            )

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
                        edgecolor="black", facecolor=color, linewidths=1, alpha=0.3, zsort = 'min'
                    )
                    ax.add_collection3d(cube)

            legend_handles = [
                plt.Line2D([0], [0], color=cmap(i), lw=3, label=f"Cluster {lab}")
                for i, lab in enumerate(unique_labels)
            ]
            ax.legend(handles=legend_handles, loc="upper left")

        else:
            color = "blue"
            for interval3d in intervals:
                x_lower, x_upper = interval3d[0, 0], interval3d[0, 1]
                y_lower, y_upper = interval3d[1, 0], interval3d[1, 1]
                z_lower, z_upper = interval3d[2, 0], interval3d[2, 1]

                faces = Interval3d.get_cube_faces(x_lower, x_upper, y_lower, y_upper, z_lower, z_upper)
                cube = Poly3DCollection(
                    faces, 
                    edgecolor="black", facecolor=color, linewidths=1, alpha=0.3, zsort = 'min'
                )
                ax.add_collection3d(cube)
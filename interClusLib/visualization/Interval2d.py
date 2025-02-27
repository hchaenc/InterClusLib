import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Interval2d:

    def draw_interval_squares(ax, intervals, labels=None):
        """
        Draw each 1D interval [lower, upper] as a square on the given matplotlib axes (ax).
        The bottom-left of each square is (lower, lower), and the top-right is (upper, upper).

        :param ax: A matplotlib axes object where the squares will be drawn.
        :param intervals: shape (n_samples, 2) or (n_samples, 1, 2)
                        Each row is [lower, upper] for a 1D interval.
        :param labels: Optional array of shape (n_samples,) for cluster labels or categories.
        """
        if intervals.ndim == 3 and intervals.shape[1] == 1:
            intervals = intervals.reshape(-1, 2)
        else:
            raise ValueError(f"Unsupported intervals shape: {intervals.shape}. Expected (n_samples, 1, 2) or similar.")

        if labels is not None:
            # Use different colors for each label
            unique_labels = np.unique(labels)
            cmap = plt.cm.get_cmap('viridis', len(unique_labels))

            for idx, lab in enumerate(unique_labels):
                mask = (labels == lab)
                sub_intervals = intervals[mask]
                color = cmap(idx)
                for [lower, upper] in sub_intervals:
                    width = upper - lower
                    height = upper - lower
                    rect = patches.Rectangle(
                        (lower, lower), width, height,
                        # linewidth=1, edgecolor="black", facecolor=color, alpha=0.5
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)

            # Create a legend using colored patches
            legend_handles = [
                patches.Patch(color=cmap(i), label=f"Cluster {lab}")
                for i, lab in enumerate(unique_labels)
            ]
            ax.legend(handles=legend_handles)

        else:
            # All intervals use the same color if no labels are provided
            color = "blue"
            for [lower, upper] in intervals:
                width = upper - lower
                height = upper - lower
                rect = patches.Rectangle(
                    (lower, lower), width, height,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)

    def draw_interval_rectangles(ax, intervals, labels=None):
        """
        Draw each 2D interval [lower, upper] as a rectangle on the given matplotlib axes (ax).
        """
        # Shape check
        if intervals.ndim != 3 or intervals.shape[1] != 2 or intervals.shape[2] != 2:
            raise ValueError(
                "Expected intervals to have shape (n_samples, 2, 2). "
                f"Got {intervals.shape} instead."
            )

        if labels is not None:
            unique_labels = np.unique(labels)
            # Use a colormap with as many colors as unique labels
            cmap = plt.cm.get_cmap('viridis', len(unique_labels))

            for idx, lab in enumerate(unique_labels):
                mask = (labels == lab)
                sub_intervals = intervals[mask]
                color = cmap(idx)
                for rect_data in sub_intervals:
                    x_lower, x_upper = rect_data[0, 0], rect_data[0, 1]
                    y_lower, y_upper = rect_data[1, 0], rect_data[1, 1]
                    
                    width = x_upper - x_lower
                    height = y_upper - y_lower
                    
                    rect = patches.Rectangle(
                        (x_lower, y_lower), width, height,
                        linewidth=2, edgecolor=color, facecolor='none'
                    )
                    ax.add_patch(rect)

            # Create a legend using colored patches
            legend_handles = [
                patches.Patch(color=cmap(i), label=f"Cluster {lab}")
                for i, lab in enumerate(unique_labels)
            ]
            ax.legend(handles=legend_handles)

        else:
            # All intervals use the same color if no labels are provided
            color = "blue"
            for rect_data in intervals:
                x_lower, x_upper = rect_data[0, 0], rect_data[0, 1]
                y_lower, y_upper = rect_data[1, 0], rect_data[1, 1]
                
                width = x_upper - x_lower
                height = y_upper - y_lower
                
                rect = patches.Rectangle(
                    (x_lower, y_lower), width, height,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
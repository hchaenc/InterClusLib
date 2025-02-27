import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

class Dendrogram:

    def plot_dendrogram(model, **kwargs):
        """
        Create linkage matrix from the children_ and distances_ of a fitted AgglomerativeClustering,
        then plot dendrogram using scipy.cluster.hierarchy.dendrogram.
        
        :param model: A fitted AgglomerativeClustering object with compute_distances=True
        :param kwargs: Additional args passed to scipy.cluster.hierarchy.dendrogram
        """

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # linkage_matrix shape = (n_samples-1, 4)
        # columns: [child1, child2, distance, cluster_size]
        
        dendrogram(linkage_matrix, **kwargs)
import pandas as pd
import numpy as np
import os

class IntervalMetrics:
    """
    A collection of similarity/distance metrics for multi-dimensional interval data.
    """

    # ========== 1. Single-Dimension Interval Similarity Measures ==========
    @staticmethod
    def jaccard_similarity(interval1, interval2):
        """
        Calculate the Jaccard similarity between two interval values.

        Parameters:
            interval1 (np.array): Interval values 1
            interval2 (np.array): Interval values 2

        Returns:
            float: Jaccard similarity between the two interval values.
        """
        # calculate the intersection of the two intervals
        intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
        # calculate the union of the two intervals
        if intersection > 0:
            union = (interval1[1] - interval1[0]) + (interval2[1] - interval2[0]) - intersection
        else:
            union = (interval1[1] - interval1[0]) + (interval2[1] - interval2[0])
        # calculate the Jaccard similarity
        jaccard = intersection / union

        return jaccard

    @staticmethod
    def dice_similarity(interval1,interval2):
        """
        Calculate the Dice similarity between two interval values.

        Parameters:
            interval1 (np.array): Interval values 1
            interval2 (np.array): Interval values 2

        Returns:
            float: Dice similarity between the two interval values.
        """
        # calculate the intersection of the two intervals
        intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
        # calculate the sum of the two intervals
        sum_intervals = interval1[1] - interval1[0] + interval2[1] - interval2[0]
        # calculate the Dice similarity
        dice = 2 * intersection / sum_intervals

        return dice
    
    @staticmethod
    def bidrectional_subset_similarity(interval1, interval2, operator):
        """
        Calculate the bidirectional subset similarity between two interval values.

        Parameters:
            interval1 (np.array): Interval values 1
            interval2 (np.array): Interval values 2
            option (str): Option to specify the type of bidirectional subset similarity to calculate.
                            1. 'minimum': Minimum bidirectional subset similarity.
                            2. 'product': Product bidirectional subset similarity.
            
        Returns:
            float: Bidirectional subset similarity between the two interval values.
        """
        # calculate the intersection of the two intervals
        intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
        # calculate the non-overlapping regions of the intervals
        non_overlap_a_b = max(0, (interval1[1] - interval1[0]) - intersection)
        non_overlap_b_a = max(0, (interval2[1] - interval2[0]) - intersection)

        # calculate the reciprocal subsethood
        reciprocal_subsethood_a_b = intersection / (intersection + non_overlap_a_b)
        reciprocal_subsethood_b_a = intersection / (intersection + non_overlap_b_a)

        # choose t-norm operator
        if operator == 'minimum':
            bidirectional_subset_similarity = min(reciprocal_subsethood_a_b, reciprocal_subsethood_b_a)
        elif operator == 'product':
            bidirectional_subset_similarity = reciprocal_subsethood_a_b * reciprocal_subsethood_b_a
        
        return bidirectional_subset_similarity

    @staticmethod
    def generalized_jaccard_similarity(interval1, interval2):
        """
        Calculate the generalized Jaccard similarity between two interval values.

        Parameters:
            interval1 (np.array): Interval values 1
            interval2 (np.array): Interval values 2

        Returns:
            float: Generalized Jaccard similarity between the two interval values.
        """
        # calculate the intersection of the two intervals
        intersection = max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))
        # calculate the union of the two intervals
        union = (interval1[1] - interval1[0]) + (interval2[1] - interval2[0]) - intersection
        # calculate the distance between the two intervals
        distance = max(0, max(interval1[0], interval2[0]) - min(interval1[1], interval2[1]))
        # calculate the domain of the two intervals
        domain = max(interval1[1], interval2[1]) - min(interval1[0], interval2[0])

        generalized_jaccard = 0.5 * (intersection / union + 1 - distance / domain)

        return generalized_jaccard
    
    # ========== 2. Single-Dimension Interval Distance Measures ==========
    @staticmethod
    def hausdorff_distance(interval1, interval2, epsilon=0):
        """
        Calculate the Hausdorff distance between two interval values.

        Parameters:
            interval1 (np.array): Interval value 1.
            interval2 (np.array): Interval value 2.

        Returns:
            float: Hausdorff distance between the two interval values.
        """
        # calculate the Hausdorff distance between two interval values
        diff2 = interval2[1] - interval1[1]
        diff1 = interval2[0] - interval1[0]

        # calculate the Hausdorff distance
        if np.sign(diff1) == np.sign(diff2):
            delta_min = min(abs(diff1), abs(diff2))
        else:
            delta_min = epsilon

        delta_max = max(abs(diff2), abs(diff1))

        mean_metric = (delta_min + delta_max) / 2
        return mean_metric   

    @staticmethod
    def range_euclidean_distance(interval1, interval2):
        """
        Calculate the Euclidean distance between the range of two interval values.

        Parameters:
            interval1 (np.array): Interval value 1.
            interval2 (np.array): Interval value 2.

        Returns:
            float: Euclidean distance between the range of the two interval values.
        """
        # calculate the Euclidean distance between the range of two interval values
        range1 = interval1[0] - interval2[0]
        range2 = interval1[1] - interval2[1]

        return np.sqrt(range1**2 + range2**2)

    @staticmethod
    def manhattan_distance(interval1, interval2):
        """
        Calculate the Manhattan distance between two interval values.

        Parameters:
            interval1 (np.array): Interval value 1.
            interval2 (np.array): Interval value 2.

        Returns:
            float: Manhattan distance between the two interval values.
        """
        # calculate the Manhattan distance between two interval values
        diff2 = interval2[1] - interval1[1]
        diff1 = interval2[0] - interval1[0]

        return abs(diff1) + abs(diff2)
    
    # ========== 3. Multi-Dimension Interval Measures ==========
    @classmethod
    def jaccard_similarity_md(cls, interval_a, interval_b, aggregate="mean"):
        """
        Multi-dimensional jaccard similarity.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :param aggregate: how to combine results (e.g., "mean", "min", "prod")
        :return: scalar similarity (usually in [0,1])
        """
        n_dims = interval_a.shape[0]
        sims = []
        for d in range(n_dims):
            sim_1d = cls.jaccard_similarity(interval_a[d], interval_b[d])
            sims.append(sim_1d)

        if aggregate == "mean":
            return np.mean(sims)
        elif aggregate == "min": 
            return np.min(sims)
        elif aggregate == "prod":  
            return np.prod(sims)
        else:
            raise ValueError(f"Unsupported aggregate method: {aggregate}")

    @classmethod
    def dice_similarity_md(cls, interval_a, interval_b, aggregate="mean"):
        """
        Multi-dimensional dice similarity.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :param aggregate: how to combine results (e.g., "mean", "min", "prod")
        :return: scalar similarity (usually in [0,1])
        """
        n_dims = interval_a.shape[0]
        sims = []
        for d in range(n_dims):
            sim_1d = cls.dice_similarity(interval_a[d], interval_b[d])
            sims.append(sim_1d)

        if aggregate == "mean":
            return np.mean(sims)
        elif aggregate == "min": 
            return np.min(sims)
        elif aggregate == "prod":  
            return np.prod(sims)
        else:
            raise ValueError(f"Unsupported aggregate method: {aggregate}")
    
    @classmethod
    def bidrectional_subset_similarity_md(cls, interval_a, interval_b, aggregate="mean"):
        """
        Multi-dimensional bidrectional subset similarity.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :param aggregate: how to combine results (e.g., "mean", "min", "prod")
        :return: scalar similarity (usually in [0,1])
        """
        n_dims = interval_a.shape[0]
        sims = []
        for d in range(n_dims):
            sim_1d = cls.bidrectional_subset_similarity(interval_a[d], interval_b[d])
            sims.append(sim_1d)

        if aggregate == "mean":
            return np.mean(sims)
        elif aggregate == "min": 
            return np.min(sims)
        elif aggregate == "prod":  
            return np.prod(sims)
        else:
            raise ValueError(f"Unsupported aggregate method: {aggregate}")

    @classmethod
    def generalized_jaccard_similarity_md(cls, interval_a, interval_b, aggregate="mean"):
        """
        Multi-dimensional generalized jaccard similarity.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :param aggregate: how to combine results (e.g., "mean", "min", "prod")
        :return: scalar similarity (usually in [0,1])
        """
        n_dims = interval_a.shape[0]
        sims = []
        for d in range(n_dims):
            sim_1d = cls.generalized_jaccard_similarity(interval_a[d], interval_b[d])
            sims.append(sim_1d)

        if aggregate == "mean":
            return np.mean(sims)
        elif aggregate == "min": 
            return np.min(sims)
        elif aggregate == "prod":  
            return np.prod(sims)
        else:
            raise ValueError(f"Unsupported aggregate method: {aggregate}")
        
    @classmethod
    def hausdorff_distance_md(cls, interval_a, interval_b, aggregate="mean"):
        """
        Multi-dimensional Hausdorff distance.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :param aggregate: how to combine results across dimensions
        :return: scalar distance
        """
        n_dims = interval_a.shape[0]
        distances = []
        for d in range(n_dims):
            dist_1d = IntervalMetrics.hausdorff_distance(interval_a[d], interval_b[d])
            distances.append(dist_1d)

        if aggregate == "mean":
            return np.mean(distances)
        elif aggregate == "sum":
            return np.sum(distances)
        elif aggregate == "max":
            return np.max(distances)
        else:
            raise ValueError(f"Unsupported aggregate method: {aggregate}")
    
    @classmethod
    def range_euclidean_distance_md(cls, interval_a, interval_b, aggregate="mean"):
        """
        Multi-dimensional range euclidean distance.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :param aggregate: how to combine results across dimensions
        :return: scalar distance
        """
        n_dims = interval_a.shape[0]
        distances = []
        for d in range(n_dims):
            dist_1d = IntervalMetrics.range_euclidean_distance(interval_a[d], interval_b[d])
            distances.append(dist_1d)

        if aggregate == "mean":
            return np.mean(distances)
        elif aggregate == "sum":
            return np.sum(distances)
        elif aggregate == "max":
            return np.max(distances)
        else:
            raise ValueError(f"Unsupported aggregate method: {aggregate}")
    
    @classmethod
    def manhattan_distance_md(cls, interval_a, interval_b, aggregate="mean"):
        """
        Multi-dimensional manhattan distance.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :param aggregate: how to combine results across dimensions
        :return: scalar distance
        """
        n_dims = interval_a.shape[0]
        distances = []
        for d in range(n_dims):
            dist_1d = IntervalMetrics.manhattan_distance(interval_a[d], interval_b[d])
            distances.append(dist_1d)

        if aggregate == "mean":
            return np.mean(distances)
        elif aggregate == "sum":
            return np.sum(distances)
        elif aggregate == "max":
            return np.max(distances)
        else:
            raise ValueError(f"Unsupported aggregate method: {aggregate}")
    
    # ========== 4. Similarity <-> Distance Conversion ==========
    @staticmethod
    def sim_to_dist(sim_value, mode="1_minus"):
        """
        Converts similarity to distance.

        :param sim_value: a similarity value in [0,1] (commonly)
        :param mode: "1_minus" means distance = 1 - similarity
        :return: scalar distance
        """
        if mode == "1_minus":
            return 1 - sim_value
        else:
            raise ValueError(f"Unsupported conversion mode: {mode}")

    @staticmethod
    def dist_to_sim(dist_value, mode="1_over_1_plus"):
        """
        Converts distance to similarity.

        :param dist_value: a non-negative distance
        :param mode: "1_over_1_plus" means similarity = 1 / (1 + distance)
        :return: scalar similarity
        """
        if mode == "1_over_1_plus":
            return 1 / (1 + dist_value)
        else:
            raise ValueError(f"Unsupported conversion mode: {mode}")
    
    # ========== 5. Compute Distance/Similarity Matrices ==========
    @classmethod
    def pairwise_similarity(cls, intervals, metric="jaccard", aggregate="mean"):
        """
        Computes an (n_samples, n_samples) similarity matrix.

        :param intervals: array of shape (n_samples, n_dims, 2)
        :param metric: which metric to use ("jaccard", etc.)
        :param aggregate: how to combine dimension-wise results ("mean", "min", "prod", etc.)
        :return: (n_samples, n_samples) similarity matrix
        """
        n_samples = intervals.shape[0]
        sim_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if metric == "jaccard":
                    sim_matrix[i, j] = IntervalMetrics.jaccard_similarity_md(
                        intervals[i], intervals[j], aggregate=aggregate)
                elif metric == "dice":
                    sim_matrix[i, j] = IntervalMetrics.dice_similarity_md(
                        intervals[i], intervals[j], aggregate=aggregate)
                elif metric == "bidrectional":
                    sim_matrix[i, j] = IntervalMetrics.bidrectional_subset_similarity_md(
                        intervals[i], intervals[j], aggregate=aggregate)
                elif metric == "generalized_jaccard":
                    sim_matrix[i, j] = IntervalMetrics.generalized_jaccard_similarity_md(
                        intervals[i], intervals[j], aggregate=aggregate)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
        return sim_matrix

    @classmethod
    def pairwise_distance(cls, intervals, metric="hausdorff", aggregate="mean"):
        """
        Computes an (n_samples, n_samples) distance matrix.

        :param intervals: array of shape (n_samples, n_dims, 2)
        :param metric: which metric to use ("midpoint", "haussdorf", etc.)
        :param aggregate: how to combine dimension-wise results ("mean", "sum", etc.)
        :return: (n_samples, n_samples) distance matrix
        """
        n_samples = intervals.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if metric == "hausdorff":
                    dist_matrix[i, j] = IntervalMetrics.hausdorff_distance_md(
                        intervals[i], intervals[j], aggregate=aggregate)
                elif metric == "range_euclidean":
                    dist_matrix[i, j] = IntervalMetrics.range_euclidean_distance_md(
                        intervals[i], intervals[j], aggregate=aggregate)
                elif metric == "manhattan":
                    dist_matrix[i, j] = IntervalMetrics.manhattan_distance_md(
                        intervals[i], intervals[j], aggregate=aggregate)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
        return dist_matrix
    

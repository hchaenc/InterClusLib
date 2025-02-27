import pandas as pd
import numpy as np
import os

class IntervalMetrics:
    """
    A collection of similarity/distance metrics for multi-dimensional interval data.
    """

    # ========== 1. Single-Dimension Interval Similarity Measures ==========
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
        jaccard = intersection / union if union > 0 else 0

        return jaccard

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
    
    def bidrectional_similarity_min(interval1, interval2):
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

        bidirectional_subset_similarity = min(reciprocal_subsethood_a_b, reciprocal_subsethood_b_a)
        
        return bidirectional_subset_similarity
    
    def bidrectional_similarity_prod(interval1, interval2):
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

        bidirectional_subset_similarity = reciprocal_subsethood_a_b * reciprocal_subsethood_b_a
        
        return bidirectional_subset_similarity

    def marginal_similarity(interval1, interval2):
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

        marginal_similarity = 0.5 * (intersection / union + 1 - distance / domain)

        return marginal_similarity
    
    # ========== 2. Single-Dimension Interval Distance Measures ==========
    def hausdorff_distance(interval1, interval2):
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

        return max(abs(diff2), abs(diff1))   

    def euclidean_distance(interval1, interval2):
        """
        Calculate the Euclidean distance between the range of two interval values.

        Parameters:
            interval1 (np.array): Interval value 1.
            interval2 (np.array): Interval value 2.

        Returns:
            float: Euclidean distance between the range of the two interval values.
        """
        # calculate the Euclidean distance between the range of two interval values
        diff1 = interval1[0] - interval2[0]
        diff2 = interval1[1] - interval2[1]

        return np.sqrt(diff1**2 + diff2**2)

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
    def jaccard_sim_md(interval_a, interval_b):
        """
        Multi-dimensional jaccard similarity.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :return: scalar distance
        """
        n_dims = interval_a.shape[0]
        sim = []
        for d in range(n_dims):
            sim_1d = IntervalMetrics.jaccard_similarity(interval_a[d], interval_b[d])
            sim.append(sim_1d)

        return np.mean(sim)

    def dice_sim_md(interval_a, interval_b):
        """
        Multi-dimensional dice similarity.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :return: scalar distance
        """
        n_dims = interval_a.shape[0]
        sim = []
        for d in range(n_dims):
            sim_1d = IntervalMetrics.dice_similarity(interval_a[d], interval_b[d])
            sim.append(sim_1d)

        return np.mean(sim)
    
    def bidrectional_sim_min_md(interval_a, interval_b):
        """
        Multi-dimensional bidrectional subset similarity.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :return: scalar distance
        """
        n_dims = interval_a.shape[0]
        sim = []
        for d in range(n_dims):
            sim_1d = IntervalMetrics.bidrectional_similarity_min(interval_a[d], interval_b[d])
            sim.append(sim_1d)

        return np.mean(sim)
    
    def bidrectional_sim_prod_md(interval_a, interval_b):
        """
        Multi-dimensional bidrectional subset similarity.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :return: scalar distance
        """
        n_dims = interval_a.shape[0]
        sim = []
        for d in range(n_dims):
            sim_1d = IntervalMetrics.bidrectional_similarity_prod(interval_a[d], interval_b[d])
            sim.append(sim_1d)

        return np.mean(sim)

    def marginal_sim_md(interval_a, interval_b):
        """
        Multi-dimensional generalized jaccard similarity.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :return: scalar distance
        """
        n_dims = interval_a.shape[0]
        sim = []
        for d in range(n_dims):
            sim_1d = IntervalMetrics.marginal_similarity_similarity(interval_a[d], interval_b[d])
            sim.append(sim_1d)
        
        return np.mean(sim)
        
    def hausdorff_distance_md(interval_a, interval_b):
        """
        Multi-dimensional Hausdorff distance.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :return: scalar distance
        """
        n_dims = interval_a.shape[0]
        distances = []
        for d in range(n_dims):
            dist_1d = IntervalMetrics.hausdorff_distance(interval_a[d], interval_b[d])
            distances.append(dist_1d)
        
        return np.sum(distances)
    
    def euclidean_distance_md(interval_a, interval_b):
        """
        Multi-dimensional range euclidean distance.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :return: scalar distance
        """
        n_dims = interval_a.shape[0]
        distances = []
        for d in range(n_dims):
            dist_1d = IntervalMetrics.euclidean_distance(interval_a[d], interval_b[d])
            distances.append(dist_1d * dist_1d)

        return np.sqrt(np.sum(distances))
    
    def manhattan_distance_md(interval_a, interval_b):
        """
        Multi-dimensional manhattan distance.

        :param interval_a: shape (n_dims, 2)
        :param interval_b: shape (n_dims, 2)
        :return: scalar distance
        """
        n_dims = interval_a.shape[0]
        distances = []
        for d in range(n_dims):
            dist_1d = IntervalMetrics.manhattan_distance(interval_a[d], interval_b[d])
            distances.append(dist_1d)

        return np.sum(distances)
    
    # ========== 4. Similarity <-> Distance Conversion ==========
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
    def pairwise_similarity(intervals, metric="jaccard"):
        """
        Computes an (n_samples, n_samples) similarity matrix.

        :param intervals: array of shape (n_samples, n_dims, 2)
        :param metric: which metric to use ("jaccard", etc.)
        :return: (n_samples, n_samples) similarity matrix
        """
        similarity_funcs = IntervalMetrics.get_similarity_funcs_md()

        if metric not in similarity_funcs:
            valid_metrics = ", ".join(similarity_funcs.keys())
            raise ValueError(f"Unsupported metric: {metric}. Available options: {valid_metrics}")

        similarity_func = similarity_funcs[metric]

        n_samples = intervals.shape[0]
        sim_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                sim_matrix[i, j] = similarity_func(intervals[i], intervals[j])

        return sim_matrix

    def pairwise_distance(intervals, metric="hausdorff"):
        """
        Computes an (n_samples, n_samples) distance matrix.

        :param intervals: array of shape (n_samples, n_dims, 2)
        :param metric: which metric to use ("hausdorff", "euclidean", "manhattan").
        :return: (n_samples, n_samples) distance matrix.
        """
        distance_funcs = IntervalMetrics.get_distance_funcs_md()

        if metric not in distance_funcs:
            valid_metrics = ", ".join(distance_funcs.keys())
            raise ValueError(f"Unsupported metric: {metric}. Available options: {valid_metrics}")

        distance_func = distance_funcs[metric]

        n_samples = intervals.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                dist_matrix[i, j] = distance_func(intervals[i], intervals[j])

        return dist_matrix
    
    def cross_similarity(self, intervals_a, intervals_b, metric="jaccard"):
        """
        Computes a (M, N) cross-similarity matrix between two sets of interval data.

        :param intervals_a: array of shape (M, n_dims, 2) - First set of intervals.
        :param intervals_b: array of shape (N, n_dims, 2) - Second set of intervals.
        :param metric: which similarity metric to use ("jaccard", "dice", etc.).
        :return: (M, N) similarity matrix.
        """
        similarity_funcs = IntervalMetrics.get_similarity_funcs_md()

        if metric not in similarity_funcs:
            valid_metrics = ", ".join(similarity_funcs.keys())
            raise ValueError(f"Unsupported metric: {metric}. Available options: {valid_metrics}")

        similarity_func = similarity_funcs[metric]

        m_samples = intervals_a.shape[0]
        n_samples = intervals_b.shape[0]
        sim_matrix = np.zeros((m_samples, n_samples))

        for i in range(m_samples):
            for j in range(n_samples):
                sim_matrix[i, j] = similarity_func(intervals_a[i], intervals_b[j])

        return sim_matrix
    
    def cross_distance(self, intervals_a, intervals_b, metric="hausdorff"):
        """
        Computes a (M, N) cross-distance matrix between two sets of interval data.

        :param intervals_a: array of shape (M, n_dims, 2) - First set of intervals.
        :param intervals_b: array of shape (N, n_dims, 2) - Second set of intervals.
        :param metric: which metric to use ("hausdorff", "euclidean", "manhattan").
        :return: (M, N) distance matrix.
        """
        distance_funcs = IntervalMetrics.get_distance_funcs_md()

        if metric not in distance_funcs:
            valid_metrics = ", ".join(distance_funcs.keys())
            raise ValueError(f"Unsupported metric: {metric}. Available options: {valid_metrics}")

        distance_func = distance_funcs[metric]

        m_samples = intervals_a.shape[0]
        n_samples = intervals_b.shape[0]
        dist_matrix = np.zeros((m_samples, n_samples))

        for i in range(m_samples):
            for j in range(n_samples):
                dist_matrix[i, j] = distance_func(intervals_a[i], intervals_b[j])

        return dist_matrix
    
    @classmethod
    def get_similarity_funcs(cls):
        return {
            "jaccard": cls.jaccard_similarity,
            "dice": cls.dice_similarity,
            "bidirectional_min": cls.bidrectional_similarity_min,
            "bidirectional_prod": cls.bidrectional_similarity_prod,
            "marginal": cls.marginal_similarity,
        }

    @classmethod
    def get_distance_funcs(cls):
        return {
            "hausdorff": cls.hausdorff_distance,
            "manhattan": cls.manhattan_distance,
            "euclidean": cls.euclidean_distance,
        }

    @classmethod
    def get_similarity_funcs_md(cls):
        return {
            "jaccard": cls.jaccard_sim_md,
            "dice": cls.dice_sim_md,
            "bidirectional_min": cls.bidrectional_sim_min_md,
            "bidirectional_prod": cls.bidrectional_sim_prod_md,
            "marginal": cls.marginal_sim_md,
        }

    @classmethod
    def get_distance_funcs_md(cls):
        return {
            "hausdorff": cls.hausdorff_distance_md,
            "manhattan": cls.manhattan_distance_md,
            "euclidean": cls.euclidean_distance_md,
        }

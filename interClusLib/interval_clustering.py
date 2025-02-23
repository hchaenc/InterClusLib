import pandas as pd
import numpy as np
from numpy import linalg, subtract, power, exp, meshgrid, zeros, outer
from numpy.random import RandomState
from warnings import warn
from math import sqrt
import os
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from interClusLib.similarity_distance import IntervalMetrics
from collections import defaultdict

class IntervalKMeans:
    """
    A custom K-Means clustering for interval data
    """

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, distance_func= 'euclidean',random_state=42):
        """
        :param n_clusters: number of clusters
        :param max_iter: maximum number of iterations
        :param tol: tolerance for convergence
        :param random_state: random seed
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = np.random.RandomState(random_state)

        sim_funcs_md = IntervalMetrics.get_similarity_funcs_md()
        dis_funcs_md = IntervalMetrics.get_distance_funcs_md()

        if distance_func in sim_funcs_md:
            self.distance_function = sim_funcs_md[distance_func]
            self.isSim = True
        elif distance_func in dis_funcs_md:
            self.distance_function = dis_funcs_md[distance_func]
            self.isSim = False
        else:
            valid_funcs = ", ".join(list(sim_funcs_md.keys()) + list(dis_funcs_md.keys()))
            raise ValueError(f"Invalid distance function '{distance_func}'. Available options: {valid_funcs}")

    def _init_centroids(self, intervals):
        """
        Initialize cluster centroids by randomly picking samples from 'intervals'.
        intervals: shape (n_samples, n_dims, 2)
        """
        n_samples = intervals.shape[0]
        # randomly choose k distinct samples as initial centroids
        indices = self.random_state.choice(n_samples, self.n_clusters, replace=False)
        centroids = intervals[indices].copy()
        return centroids

    def _compute_centroid(self, intervals_in_cluster):
        """
        Compute the centroid of intervals in one cluster.
        intervals_in_cluster: shape (k, n_dims, 2)
        """
        # mean of lower bounds, mean of upper bounds dimension-wise
        return np.mean(intervals_in_cluster, axis=0)
    
    def _assign_clusters(self, intervals, centroids):
        """
        Assign each sample in 'intervals' to the nearest centroid using 'distance_func'.
        """
        n_samples = intervals.shape[0]
        labels = np.zeros(n_samples, dtype=np.int32)

        for i in range(n_samples):
            # compute distance to each centroid
            dists = [self.distance_function(intervals[i], c) for c in centroids]
            if self.isSim:
                labels[i] = np.argmax(dists)  # 相似性：选择值最大的 centroid
            else:
                labels[i] = np.argmin(dists)
        return labels
    
    def fit(self, intervals):
        """
        intervals: shape (n_samples, n_dims, 2)
        distance_func: function that takes (interval_a, interval_b) and returns a scalar distance
        """
        # 1. Initialize centroids
        centroids = self._init_centroids(intervals)

        for iteration in range(self.max_iter):
            # 2. Assign clusters
            labels = self._assign_clusters(intervals, centroids)

            # 3. Compute new centroids
            new_centroids = []
            for k in range(self.n_clusters):
                cluster_points = intervals[labels == k]
                if len(cluster_points) > 0:
                    centroid_k = self._compute_centroid(cluster_points)
                else:
                    # if no points assigned, re-initialize or handle it in some way
                    centroid_k = centroids[k]
                new_centroids.append(centroid_k)
            new_centroids = np.array(new_centroids)

            # 4. Check for convergence (centroid shift)
            shift = np.sum((centroids - new_centroids)**2)
            centroids = new_centroids
            if shift < self.tol:
                break

        # save final centroids and labels
        self.centroids_ = centroids
        self.labels_ = labels

    def predict(self, intervals):
        """
        Assign new data points to the closest cluster.
        """
        labels = self._assign_clusters(intervals, self.centroids_)
        return labels
    
class IntervalAgglomerativelustering:
    """
    An Agglomerative (Hierarchical) Clustering for interval data (n_dims, 2).
    Uses a precomputed distance matrix from a custom distance function.
    """
    distance_funcs = {"hausdorff", "euclidean", "manhattan"}
    similarity_funcs = {"jaccard", "dice", "bidrectional", "generalized_jaccard"}

    def __init__(self, n_clusters=2, linkage='average', distance_func = 'euclidean'):
        """
        :param n_clusters: int, number of clusters to find (you can also set distance_threshold instead).
        :param linkage: str, linkage criterion ('complete', 'average', 'single').
        :param distance_func: a function(interval_a, interval_b) -> distance (scalar).
        :param aggregate: str, how to combine distance across dimensions if needed (e.g., 'mean', 'sum', 'max').
        """
        self.n_clusters = n_clusters
        self.linkage = linkage

        self.model_ = None
        self.labels_ = None
        self.distance_func = distance_func

        if self.distance_func in self.distance_funcs:
            self.isSim = False
        elif self.distance_func in self.similarity_funcs:
            self.isSim = True
        else:
            raise ValueError(f"Unsupported metric: {self.distance_func}")
    
    def compute_distance_matrix(self, intervals):
        if self.isSim:
            dist_matrix = IntervalMetrics.pairwise_similarity(
                intervals, 
                metric=self.distance_func
            )
            dist_matrix = 1 - dist_matrix
        else:
            dist_matrix = IntervalMetrics.pairwise_distance(
                intervals, 
                metric=self.distance_func
            )
        return dist_matrix

    def fit(self, intervals, convert_mode = None):
        """
        :param intervals: shape (n_samples, n_dims, 2)
        """
        dist_matrix = self.compute_distance_matrix(intervals)

        self.model_ = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric='precomputed',  # or metric='precomputed' in newer sklearn
            linkage=self.linkage,
            compute_distances=True
        )
        self.model_.fit(dist_matrix)

        self.labels_ = self.model_.labels_
        
    def fit_predict(self, intervals):
        self.fit(intervals)
        return self.labels_

    def get_labels(self):
        if self.labels_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.labels_

class IntervalFuzzyCMeans:
    """
    Interval Fuzzy C-Means (IFCM & IFCMADC) for interval-valued data (n_samples, n_dims, 2).

    Attributes
    ----------
    - n_clusters : int
        Number of clusters (C).
    - m : float
        Fuzzifier (typically = 2).
    - max_iter : int
        Maximum number of iterations.
    - tol : float
        Convergence threshold for membership changes or objective.
    - adaptive_weights : bool
        If True, use IFCMADC method with adaptive weights k_i^j.
    - distance_method : str
        Which distance to use for D(x_k, c_i). Example: "sum_squares" / "hausdorff".
    - U : np.ndarray of shape (n_samples, n_clusters)
        Fuzzy membership matrix.
    - centers_a, centers_b : np.ndarray of shape (n_clusters, n_dims)
        The lower and upper bounds for each cluster center.
    - k : np.ndarray of shape (n_clusters, n_dims)
        Adaptive weights, only used if adaptive_weights=True.
    - objective_ : float
        The value of the objective function after final iteration.
    """

    def __init__(self,
                 n_clusters=2,
                 m=2.0,
                 max_iter=100,
                 tol=1e-5,
                 adaptive_weights=False,
                 distance_func = "euclidean",
                 ):
        """
        Parameters
        ----------
        n_clusters : int
            Number of clusters (C).
        m : float
            Fuzzifier, controlling fuzziness. Commonly m=2.
        max_iter : int
            Maximum number of iterations to run.
        tol : float
            Convergence threshold for membership matrix changes or objective function.
        adaptive_weights : bool
            If True, uses IFCMADC with adaptive weights k_i^j.
            If False, uses basic IFCM.
        metric: str
            Distance Function
        """
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.adaptive_weights = adaptive_weights

        self.U = None
        self.centers_a = None  # shape (n_clusters, n_dims)
        self.centers_b = None  # shape (n_clusters, n_dims)
        self.k = None          # shape (n_clusters, n_dims) if adaptive_weights=True
        self.objective_ = None

        sim_funcs = IntervalMetrics.get_similarity_funcs()
        dis_funcs = IntervalMetrics.get_distance_funcs()

        if distance_func in sim_funcs:
            self.distance_function = sim_funcs[distance_func]
            self.isSim = True
        elif distance_func in dis_funcs:
            self.distance_function = dis_funcs[distance_func]
            self.isSim = False
        else:
            valid_funcs = ", ".join(list(sim_funcs.keys()) + list(dis_funcs.keys()))
            raise ValueError(f"Invalid distance function '{distance_func}'. Available options: {valid_funcs}")

    def _init_membership(self, n_samples):
        """
        Randomly initialize membership matrix U (shape (n_samples, n_clusters)).
        Each row sums to ~1, but not strictly required for random init.
        """
        U = np.random.rand(n_samples, self.n_clusters)
        U = U / U.sum(axis=1, keepdims=True)
        return U

    def _update_centers(self, intervals):
        """
        Update cluster centers (a_i^j, b_i^j) using the formula:
        a_i^j = ( sum_{k=1 to N} (u_{ik}^m * a_k^j ) ) / ( sum_{k=1 to N} (u_{ik}^m) )
        b_i^j = ( sum_{k=1 to N} (u_{ik}^m * b_k^j ) ) / ( sum_{k=1 to N} (u_{ik}^m) )
        intervals: shape (n_samples, n_dims, 2)
        """
        n_samples, n_dims, _ = intervals.shape
        U_m = self.U ** self.m  # shape (n_samples, n_clusters)

        centers_a = np.zeros((self.n_clusters, n_dims))
        centers_b = np.zeros((self.n_clusters, n_dims))

        denom = U_m.sum(axis=0)  # shape (n_clusters,)

        for i in range(self.n_clusters):
            # Weighted sum of a_k^j
            for j in range(n_dims):
                numerator_a = 0.0
                numerator_b = 0.0
                for k in range(n_samples):
                    numerator_a += U_m[k, i] * intervals[k, j, 0]  # a_k^j
                    numerator_b += U_m[k, i] * intervals[k, j, 1]  # b_k^j
                centers_a[i, j] = numerator_a / (denom[i] + 1e-16)
                centers_b[i, j] = numerator_b / (denom[i] + 1e-16)

        self.centers_a = centers_a
        self.centers_b = centers_b

    def _compute_distance(self, x_k, c_i):
        """
        Compute distance between data point x_k=(a_k^j,b_k^j) and cluster center c_i=(a_i^j,b_i^j).
        If IFCMADC -> use k_i^j as adaptive weight for each dimension.

        x_k : shape (n_dims,2)
        c_i : (a_i^j, b_i^j), shape (n_dims,2)
        k_i : shape (n_dims, ) or None

        Return a scalar distance value.
        """
        distances = np.zeros(x_k.shape[0], dtype=np.float64)

        for j in range(x_k.shape[0]):
            if self.isSim:
                distances[j] = 1 - self.distance_function(x_k[j], c_i[j])
            else:
                distances[j] = self.distance_function(x_k[j], c_i[j])

        return distances

    def _update_membership(self, distances):
        """
        Update membership matrix U based on distances D(x_k, c_i).
        u_{ik} = 1 / sum_{h=1..C} ( D(x_k,c_i) / D(x_k,c_h) )^(1/(m-1))
        distances : shape (n_samples, n_clusters, n_dim), precomputed
        """
        n_samples, n_clusters, n_dims = distances.shape

        if self.adaptive_weights:
            weighted_distances = np.sum(self.k[None, :, :] * distances, axis=2)  # (n_samples, n_clusters)
        else:
            weighted_distances = np.sum(distances, axis=2)

        U = np.zeros((n_samples, n_clusters), dtype=np.float64)

        for k in range(n_samples):
            # Check if the sample has zero distance to any cluster
            zero_distance_indices = np.where(weighted_distances[k, :] == 0)[0]

            if len(zero_distance_indices) > 0:
                # Assign full membership to the nearest cluster and set others to 0
                U[k, zero_distance_indices[0]] = 1.0
                continue  # Skip normalization for this sample

            # Compute membership
            for i in range(n_clusters):
                denom_sum = np.sum([(weighted_distances[k, i] / weighted_distances[k, h]) ** (2 / (self.m - 1))
                                    for h in range(n_clusters)])
                U[k, i] = 1.0 / (denom_sum + 1e-16)  # Avoid division by zero

        return U 

    def _compute_adaptive_weights(self, distances):
        """
        Computes the adaptive weight k_i^j for each cluster and feature dimension 
        in the IFCMADC (Interval-Valued Fuzzy C-Means with Adaptive Distance Calculation) method.

        Function:
            k_i^j = [ ( Prod_{h=1..p} sum_{k} (u_{ik}^m * dist_h^2 ) ) ^(1/p) ]
                    / ( sum_{k} (u_{ik}^m * dist_j^2 ) )

        Parameters:
            distances (np.ndarray): A 3D array of shape (n_samples, n_clusters, n_dims) 
                                    representing the precomputed distance matrix D.

        Updates:
            - self.k: A 2D array of shape (n_clusters, n_dims) containing the adaptive weights.

        Notes:
            - The function utilizes the log-sum-exp trick to prevent numerical overflow when computing 
            the product term in the numerator.
            - The weights are clipped to a reasonable range [1e-3, 1e3] to prevent extreme values.
        """
        n_samples, n_clusters, n_dims = distances.shape
        eps = 1e-16  # Avoid log(0) issues

        # Compute u_{ik}^m (fuzzy membership raised to the power of m)
        U_m = self.U ** self.m  # (n_samples, n_clusters)

        # Compute sum_{k} (u_{ik})^m * D_j^2
        denom = np.sum(U_m[:, :, None] * distances, axis=0) + eps  # (n_clusters, n_dims)

        # Compute log(sum_{k} (u_{ik})^m * D_h^2) to prevent numerical overflow
        log_sums = np.sum(np.log(denom), axis=1) / n_dims  # (n_clusters,)

        # Compute log(k_i^j) and exponentiate to obtain k_i^j
        log_k = log_sums[:, None] - np.log(denom)

        # compute k_i^j
        self.k = np.exp(log_k)
        # Clip k_i^j to prevent extreme values
        self.k = np.clip(self.k, 1e-3, 1e3)

    def _compute_distances_matrix(self, intervals):
        """
        Return shape (n_samples, n_clusters) distance matrix.
        If adaptive_weights is True, then pass k_i^j in computing distance.
        """
        n_samples, n_dims, _ = intervals.shape
        distances = np.zeros((n_samples, self.n_clusters, n_dims), dtype=np.float64)
        for i in range(self.n_clusters):
            c_i = np.stack([self.centers_a[i, :], self.centers_b[i, :]], axis=1)  # shape (n_dims,2)
            # c_i[j,0]=a_i^j, c_i[j,1]=b_i^j
            for k in range(n_samples):
                x_k = intervals[k]  # shape (n_dims,2)
                distances[k, i, :] = self._compute_distance(x_k, c_i)
        return distances

    def _compute_objective(self, distances):
        """
        Computes the objective function J, which measures the clustering performance.

        Parameters:
            distances
        Returns:
            float: The objective function value J.
        Notes:
            - The objective function is defined as:
            J = sum_{i=1}^{c} sum_{k=1}^{N} (u_{ik})^m sum_{j=1}^{p} k_i^j * D_j^2(x_k, c_i)
            - If adaptive weights (k_i^j) are used, the distances are weighted accordingly.
        """
        n_samples, n_clusters, n_dims = distances.shape

        # Compute u_{ik}^m (fuzzy membership raised to the power of m)
        U_m = self.U ** self.m  # (n_samples, n_clusters)

        # Compute weighted distances if adaptive weights are enabled
        if self.adaptive_weights and hasattr(self, 'k') and self.k is not None:
            weighted_distances = np.sum(self.k[None, :, :] * distances, axis=2)  # (n_samples, n_clusters)
        else:
            weighted_distances = np.sum(distances, axis=2)  # (n_samples, n_clusters)

        # Compute the objective function J by summing the membership-weighted distances
        J = np.sum(U_m * weighted_distances)

        return J

    def fit(self, intervals):
        """
        Main iterative loop:
        1. initialize U
        2. repeat:
           a) update centers
           b) compute distances
           c) if adaptive_weights -> compute k_i^j
           d) update membership U
           e) check convergence or max_iter
        3. store final objective
        """
        n_samples, n_dims, _ = intervals.shape
        # 1. init membership
        self.U = self._init_membership(n_samples)

        prev_U = self.U.copy()
        prev_obj = None

        for iteration in range(self.max_iter):
            # a) update centers
            self._update_centers(intervals)

            # b) compute distances
            distances = self._compute_distances_matrix(intervals)

            # c) if adaptive, compute k_i^j
            if self.adaptive_weights:
                self._compute_adaptive_weights(distances)

            # d) update membership
            new_U = self._update_membership(distances)

            # e) check for convergence
            diff_U = np.abs(new_U - self.U).max()
            self.U = new_U

            # compute objective
            obj = self._compute_objective(distances)

            if prev_obj is not None and abs(obj - prev_obj) < 1e-8:
                # objective stable
                break
            if diff_U < self.tol:
                # membership stable
                break

            prev_obj = obj
            prev_U = self.U.copy()

        self.objective_ = obj
        return self

    def predict(self, intervals):
        """
        Predicts the cluster assignments for new interval-valued data.

        Parameters:
            intervals (np.ndarray): A 3D array of shape (n_samples, n_dims, 2), 
                                    where each sample consists of interval-valued features.

        Returns:
            np.ndarray: A 1D array of shape (n_samples,) containing the hard cluster assignments.
        """
        if self.U is None or self.centers_a is None or self.centers_b is None:
            raise RuntimeError("Model has not been fitted yet. Please call `fit()` before prediction.")

        distances = self._compute_distances_matrix(intervals)
        
        # Compute membership values for the new intervals
        new_U = self._update_membership(intervals, distances)

        # Return hard assignments (cluster index with the highest membership)
        return new_U  # Shape: (n_samples, n_clusters)

    def get_membership(self):
        """
        Return final membership matrix (U) after fit.
        """
        if self.U is None:
            raise RuntimeError("Model not fitted yet.")
        return self.U

    def get_centers(self):
        """
        Return the final cluster center intervals: (centers_a, centers_b).
        """
        if self.centers_a is None or self.centers_b is None:
            raise RuntimeError("Model not fitted yet.")
        return self.centers_a, self.centers_b

    def get_objective(self):
        """
        Return final objective function value J.
        """
        return self.objective_

    def get_hard_assignments(self):
        """
        Returns a 1D array of shape (n_samples,), where each element represents 
        the index of the cluster with the highest membership value for the corresponding sample.

        Raises:
            RuntimeError: If the model has not been fitted yet (U is None).
        
        Returns:
            np.ndarray: A 1D array containing the cluster assignments.
        """
        if self.U is None:
            raise RuntimeError("Model not fitted yet. U is None.")

        # np.argmax finds the index of the maximum value along axis=1 (row-wise),
        # assigning each sample to the cluster with the highest membership.
        crisp_labels = np.argmax(self.U, axis=1)
        return crisp_labels

class IntervalSOM:

    def __init__(self, 
                 x,
                 y,
                 n_dims,
                 learning_rate,
                 total_iterations,
                 sigma_init,
                 sigma_final = None,
                 decay_function = 'linear_decay_to_zero',
                 neighborhood_function = 'gaussian',
                 activation_distance = 'euclidean',
                 topology='rectangular',
                 random_seed=None,
                 sigma_decay_function='linear_interpolate'):
        """
        Initializes an Interval Self-Organizing Map (SOM).

        Parameters:
        -----------
        x : int
            Number of rows in the SOM grid.
        y : int
            Number of columns in the SOM grid.
        n_dims : int
            Number of dimensions (features) in the input data.
        sigma_init : float
            Initial neighborhood radius.
        sigma_final : float
            Final neighborhood radius.
        learning_rate : float
            Initial learning rate.
        total_iterations : int
            Total number of training iterations.
        decay_function : str or callable, optional (default='asymptotic_decay')
            The function that determines how the learning rate decreases.
        neighborhood_function : str, optional (default='gaussian')
            Defines how neighboring neurons are influenced during training.
        activation_distance : str, optional (default='euclidean')
            The distance metric used for Best Matching Unit (BMU) selection.
        topology : str, optional (default='rectangular')
            Determines the grid structure of the SOM ('rectangular' or 'hexagonal').
        random_seed : int, optional (default=None)
            Random seed for reproducibility.
        sigma_decay_function : str, optional (default='asymptotic_decay')
            The function controlling how the neighborhood radius shrinks.
        """

        if sigma_init > sqrt(x*x + y*y):
            warn('Warning: sigma might be too high ' +
                 'for the dimension of the map.')

        self.x = x
        self.y = y
        self.n_dims = n_dims
        self.sigma_init = sigma_init
        self.sigma_final = sigma_final
        self.learning_rate = learning_rate
        self.total_iterations = total_iterations
        self.topology = topology
        self.random_seed = random_seed

        # Create a random number generator for initialization and sampling
        self._rng = np.random.RandomState(self.random_seed)

        # 1) Initialize the SOM prototypes (neurons' weight vectors)
        #    Each neuron stores n_dims interval values: [lower, upper]
        #    Shape: (x, y, n_dims, 2)
        self.prototypes = self._rng.rand(x, y, n_dims, 2)
        # Ensure that lower bounds are always ≤ upper bounds
        self.prototypes = np.sort(self.prototypes, axis=-1)

        # 2) Store the (row, col) coordinates of each neuron in the SOM grid
        #    For rectangular topology, (i, j) represents its position
        #    For hexagonal topology, a slight shift is applied
        self._locations = np.zeros((x, y, 2), dtype=float)
        for i in range(x):
            for j in range(y):
                self._locations[i,j] = [i, j]
        
        # create activation_map for each sample. shape => (x, y)
        self._activation_map = np.zeros((x, y))

        if self.topology == 'hexagonal':
            if neighborhood_function in ['triangle']:
                warn('triangle neighborhood function does not ' +
                     'take in account hexagonal topology')
            # Shift even rows horizontally by 0.5 to create a hexagonal layout
            for i in range(x):
                if i % 2 == 1:
                    self._locations[i,:, 0] = i
                    self._locations[i,:, 1] = np.arange(y) + 0.5
                else:
                    self._locations[i,:, 0] = i
                    self._locations[i,:, 1] = np.arange(y)
        elif self.topology == 'rectangular':
            pass
        else:
            raise ValueError("Topology must be either 'rectangular' or 'hexagonal'.")
        
        # 4) Learning rate decay functions
        self.lr_decay_functions = {
            'inverse_decay_to_zero': self._inverse_decay_to_zero,
            'linear_decay_to_zero': self._linear_decay_to_zero,
            'asymptotic_decay': self._asymptotic_decay}
        
        # Validate and assign decay function
        if isinstance(decay_function, str):
            if decay_function not in self.lr_decay_functions:
                msg = '%s not supported. Available functions: %s'
                raise ValueError(msg % (decay_function, ', '.join(self.lr_decay_functions.keys())))

            self._learning_rate_decay_function = self.lr_decay_functions[decay_function]
        elif callable(decay_function):
            self._learning_rate_decay_function = decay_function
        else:
            raise ValueError("decay_function must be a string or a callable function.")
        
        # 5) Sigma (neighborhood radius) decay functions
        sig_decay_functions = {
            'inverse_decay_to_one': self._inverse_decay_to_one,
            'linear_decay_to_one': self._linear_decay_to_one,
            'asymptotic_decay': self._asymptotic_decay,
            'linear_interpolate': self._linear_interpolate
            
        }

        if sigma_decay_function not in sig_decay_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (sigma_decay_function, ', '.join(sig_decay_functions.keys())))

        if sigma_decay_function == 'linear_interpolate' and self.sigma_final == 0:
            msg = "Error: Using 'linear_interpolate' requires sigma_final > 0."
            raise ValueError(msg)
        self._sigma_decay_function = sig_decay_functions[sigma_decay_function]
        
        # 6) Define neighborhood influence functions
        neig_functions = {
            'gaussian': self._gaussian,
            'mexican_hat': self._mexican_hat,
            'bubble': self._bubble,
            'triangle': self._triangle
        }

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function, ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle', 'bubble'] and (divmod(sigma, 1)[1] != 0 or sigma < 1):
            warn('sigma should be an integer >=1 when triangle or bubble are used as neighborhood function')

        self._neighborhood = neig_functions[neighborhood_function]

        # 7) Define distance functions for computing the BMU (Best Matching Unit)
        sim_funcs_md = IntervalMetrics.get_similarity_funcs_md()
        dis_funcs_md = IntervalMetrics.get_distance_funcs_md()

        # Validate and assign the activation distance function
        if isinstance(activation_distance, str):
            if activation_distance in sim_funcs_md:
                self._activation_distance = sim_funcs_md[activation_distance]
                self.isSim = True
            elif activation_distance in dis_funcs_md:
                self._activation_distance = dis_funcs_md[activation_distance]
                self.isSim = False
            else:
                raise ValueError(f"'{activation_distance}' not supported. "
                    f"Available distance functions: {', '.join(dis_funcs_md.keys())}, "
                    f"Available similarity functions: {', '.join(sim_funcs_md.keys())}"
                )
        elif callable(activation_distance):
            self._activation_distance = activation_distance

        

    def random_weights_init(self, intervals):
        """
        Initializes each neuron prototype by randomly selecting an interval 
        from the input 'intervals' dataset of shape (N, n_dims, 2).

        Parameters:
        -----------
        intervals : np.ndarray
            The dataset containing interval-valued data.
            Shape: (N, n_dims, 2), where:
            - N : Number of samples
            - n_dims : Number of dimensions (features)
            - 2 : Lower and upper bounds for each dimension

        Raises:
        -------
        ValueError:
            If the input dataset does not match the expected shape (N, n_dims, 2).
        """
        n_samples, dims, two_ = intervals.shape
        if dims != self.n_dims or two_ != 2:
            raise ValueError(f"Data shape must be (N, {self.n_dims}, 2). Got {intervals.shape}.")
        # 2) Assign each neuron (i, j) a randomly chosen interval from the dataset
        for i in range(self.x):
            for j in range(self.y):
                rand_idx = self._rng.randint(n_samples)
                self.prototypes[i,j] = intervals[rand_idx]  # shape (n_dims,2)
    
    def pca_weights_init(self, intervals):
        """
        Initializes the SOM neuron prototypes using a PCA-based approach:

        Steps:
        1) For each sample's interval [a,b], compute the center: 
        c = (a + b) / 2  → Shape: (N, n_dims)
        2) Perform PCA on these centers to extract the first two principal components (PC1, PC2).
        3) Map each neuron (i,j) to a 2D coordinate in the range [-1,1]^2 → (c1, c2).
        Then, reconstruct the "center" using:
        center = mean + c1 * PC1 + c2 * PC2 in R^n_dims.
        4) For each dimension `d`, assign:
        prototypes[i, j, d] = [center_d, center_d] → Meaning the interval collapses to a single point.

        Parameters:
        -----------
        intervals : np.ndarray
            Input interval-valued data.
            Shape: (N, n_dims, 2), where:
            - N : Number of samples
            - n_dims : Number of dimensions (features)
            - 2 : Lower and upper bounds for each dimension

        Raises:
        -------
        ValueError:
            If the input dataset does not match the expected shape (N, n_dims, 2).
            If `n_dims == 1`, since PCA-based initialization requires at least 2 dimensions.
        """

        # Validate input shape
        n_samples, dims, two_ = intervals.shape
        if dims != self.n_dims or two_ != 2:
            raise ValueError(f"Data shape must be (N, {self.n_dims}, 2). Got {intervals.shape}.")

        # 1) Compute interval centers: Shape (N, n_dims)
        centers = 0.5 * (intervals[..., 0] + intervals[..., 1])  # Averaging lower and upper bounds

        # 2) Perform PCA
        # Step a) Compute mean
        mean_c = centers.mean(axis=0)  # Shape (n_dims,)

        # Step b) Compute covariance matrix
        X = centers - mean_c  # Centering data, Shape (N, n_dims)
        cov_ = np.cov(X, rowvar=False)  # Shape (n_dims, n_dims)

        # Step c) Eigen decomposition
        eigvals, eigvecs = np.linalg.eig(cov_)

        # Step d) Sort eigenvalues and pick the top 2 principal components
        idx_sorted = np.argsort(-eigvals)  # Descending order
        v1 = eigvecs[:, idx_sorted[0]]  # First principal component (Shape: n_dims,)
        v2 = eigvecs[:, idx_sorted[1]]  # Second principal component (Shape: n_dims,)

        # Ensure that PCA-based initialization is meaningful
        if self.n_dims == 1:
            raise ValueError("PCA initialization requires at least 2 dimensions. Try random initialization instead.")

        # 3) Map (i, j) positions in the SOM grid to a coordinate in [-1,1]^2 → (c1, c2)
        #    - `c1_vals` maps `i` values from 0 to (x-1) linearly into [-1,1]
        #    - `c2_vals` maps `j` values from 0 to (y-1) linearly into [-1,1]
        #    - The neuron center in feature space is reconstructed as:
        #      center = mean_c + c1 * v1 + c2 * v2

        c1_vals = np.linspace(-1, 1, self.x)  # Shape (x,)
        c2_vals = np.linspace(-1, 1, self.y)  # Shape (y,)

        for i in range(self.x):
            for j in range(self.y):
                c1 = c1_vals[i]
                c2 = c2_vals[j]

                # Compute reconstructed center in feature space
                center_ij = mean_c + c1 * v1 + c2 * v2  # Shape (n_dims,)

                # 4) Assign the same center value to both lower and upper bounds
                for d in range(self.n_dims):
                    val_d = center_ij[d]
                    self.prototypes[i, j, d, 0] = val_d  # Lower bound
                    self.prototypes[i, j, d, 1] = val_d  # Upper bound (Collapsed interval)

    # -------------- decay functions ---------------
    def _inverse_decay_to_zero(self, learning_rate, t, max_iter):
        """Learning rate decreases asymptotically to zero."""
        C = max_iter / 100.0
        return learning_rate * C / (C + t)

    def _linear_decay_to_zero(self, learning_rate, t, max_iter):
        """Learning rate decreases linearly to zero."""
        return learning_rate * (1 - t / max_iter)

    def _asymptotic_decay(self, learning_rate, t, max_iter):
        """Learning rate asymptotically decreases to 1/3 of its original value."""
        return learning_rate / (1 + t / (max_iter / 2))
    
    def _inverse_decay_to_one(self, sigma, t, max_iter):
        """Sigma decays asymptotically to 1."""
        C = (sigma - 1) / max_iter
        return sigma / (1 + (t * C))
    
    def _linear_decay_to_one(self, sigma, t, max_iter):
        """Sigma decreases linearly to 1."""
        return sigma + (t * (1 - sigma) / max_iter)
    
    def _linear_interpolate(self, sigma, t, max_iter):
        return sigma + (t/float(max_iter))*(self.sigma_final - sigma)

    # -------------- neighborhood functions -------------
    def _gaussian(self, c, sigma):
        """
        Gaussian neighborhood function centered at BMU `c`.

        Parameters:
            c (tuple): Coordinates of the BMU (row, col).
            sigma (float): Current neighborhood radius.

        Returns:
            np.ndarray: Neighborhood influence matrix.
        """
        d = 2 * sigma * sigma
        ax = np.exp(-np.power(self._locations[:, :, 0] - c[0], 2) / d)
        ay = np.exp(-np.power(self._locations[:, :, 1] - c[1], 2) / d)
        return (ax * ay)
    
    def _mexican_hat(self, c, sigma):
        """
        Mexican hat (Ricker wavelet) neighborhood function.

        Parameters:
            c (tuple): Coordinates of the BMU (row, col).
            sigma (float): Current neighborhood radius.

        Returns:
            np.ndarray: Neighborhood influence matrix.
        """
        d = 2 * sigma * sigma
        p = np.power(self._locations[:, :, 0] - c[0], 2) + np.power(self._locations[:, :, 1] - c[1], 2)
        return np.exp(-p / d) * (1 - 2 / d * p)
    
    def _bubble(self, c, sigma):
        """
        Bubble neighborhood function with constant influence within a fixed radius.

        Parameters:
            c (tuple): Coordinates of the BMU (row, col).
            sigma (float): Current neighborhood radius.

        Returns:
            np.ndarray: Neighborhood influence matrix.
        """
        ax = np.logical_and(self._locations[:, :, 0] > c[0] - sigma,
                            self._locations[:, :, 0] < c[0] + sigma)
        ay = np.logical_and(self._locations[:, :, 1] > c[1] - sigma,
                            self._locations[:, :, 1] < c[1] + sigma)
        return np.outer(ax, ay).astype(float)
    
    def _triangle(self, c, sigma):
        """
        Triangular neighborhood function.

        Parameters:
            c (tuple): Coordinates of the BMU (row, col).
            sigma (float): Current neighborhood radius.

        Returns:
            np.ndarray: Neighborhood influence matrix.
        """
        triangle_x = (-np.abs(self._locations[:, :, 0] - c[0])) + sigma
        triangle_y = (-np.abs(self._locations[:, :, 1] - c[1])) + sigma
        triangle_x[triangle_x < 0] = 0.0
        triangle_y[triangle_y < 0] = 0.0
        return np.outer(triangle_x, triangle_y)
    
    def _distance_from_weights(self, intervals):
        """
        For each sample => compute distance to each neuron => shape(N, x*y)
        """
        N = intervals.shape[0]
        dist_mat = np.zeros((N, self.x*self.y))
        idx=0
        for s in range(N):
            sample = intervals[s]
            col=0
            for i in range(self.x):
                for j in range(self.y):
                    dist_mat[s,col] = self._activation_distance(sample, self.prototypes[i,j])
                    if self.isSim:
                        dist_mat[s,col] = 1 - dist_mat[s,col]
                    col+=1
        return dist_mat
    
    def winner(self, interval):
        """
        for incremental => find best (i,j)
        """
        min_dist=1e15
        best=(0,0)
        for i in range(self.x):
            for j in range(self.y):
                dist_ij = self._activation_distance(interval, self.prototypes[i,j])
                if self.isSim:
                    dist_ij = 1 - dist_ij
                if dist_ij<min_dist:
                    min_dist=dist_ij
                    best=(i,j)
        return best
    
    def train_incremental(self, intervals, sample_mode='random', verbose=False):
        """
        Incremental (online) training for interval SOM.

        Parameters
        ----------
        data : np.ndarray
            Training data of shape (N, n_dims, 2).

        sample_mode : str, optional
            How to pick the sample each iteration.
            - 'random': pick a random sample from data (default).
            - 'sequential': pick sample in sequential order (t % N).

        verbose : bool, optional
            If True, print intermediate quantization error every 10 steps.
        """
        N = intervals.shape[0]
        seq_idx = 0  # for 'sequential' mode

        for t in range(self.total_iterations):
            # 1. Pick sample according to sample_mode
            if sample_mode == 'random':
                rand_idx = self._rng.randint(N)
                sample = intervals[rand_idx]
            elif sample_mode == 'sequential':
                sample = intervals[seq_idx]
                seq_idx = (seq_idx + 1) % N
            else:
                raise ValueError(f"Unknown sample_mode='{sample_mode}'. Use 'random' or 'sequential'.")

            # 2. Find BMU
            bmu_ij = self.winner(sample)

            # 3. Compute new learning rate
            lr_t = self._learning_rate_decay_function(self.learning_rate, t, self.total_iterations)

            # 4. Compute new sigma
            sig_t = self._sigma_decay_function(self.sigma_init, t, self.total_iterations)

            # 5. Compute neighbor matrix
            neigh = self._neighborhood(bmu_ij, sig_t)

            # 6. Update prototypes with partial "pull" on intervals
            for i in range(self.x):
                for j in range(self.y):
                    influence = neigh[i,j] * lr_t
                    if influence > 1e-9:
                        old_ij = self.prototypes[i,j]  # shape (n_dims,2)
                        delta = sample - old_ij        # shape (n_dims,2)
                        self.prototypes[i,j] = old_ij + influence * delta
                        # ensure lower <= upper
                        self.prototypes[i,j] = np.sort(self.prototypes[i,j], axis=-1)

            # (Optional) print progress
            if verbose and (t % 10 == 0):
                qe = self.quantization_error(intervals)
                print(f"[Incremental-{sample_mode}] Iter {t}, QE={qe:.4f}")
    
    def train_batch(self, intervals: np.ndarray, verbose=False):
        """
        1) for t in 0..total_iterations-1:
           - dist_mat => BMU
           - for each neuron => do WeightedMedian update
        """
        N = intervals.shape[0]
        # precompute m_i^d = (a_i^d + b_i^d)/2,  l_i^d = (b_i^d - a_i^d)/2
        intervals_m = 0.5*(intervals[...,0]+intervals[...,1]) # shape(N,n_dims)
        intervals_l = 0.5*(intervals[...,1]-intervals[...,0]) # shape(N,n_dims)
        for t in range(self.total_iterations):
            # compute sigma
            sig_t = self._sigma_decay_function(self.sigma_init, t, self.total_iterations)
            # compute dist => bmu
            dist_mat = self._distance_from_weights(intervals)
            bmu_flat = np.argmin(dist_mat, axis=1)    # shape(N,) => best neuron in flatten (i*y + j)
            # compute neighbor weight => shape(x,y,N)
            neighbor_w = np.zeros((self.x, self.y, N))
            for s in range(N):
                bf = bmu_flat[s]
                c_i = bf//self.y
                c_j = bf%self.y
                # high-level => neighbor
                nb_mat = self._neighborhood( (c_i,c_j), sig_t )  # shape(x,y)
                neighbor_w[:,:,s] = nb_mat

            # update prototypes => WeightedMedian
            self._update_prototypes_batch(intervals, intervals_m, intervals_l, neighbor_w)

            if verbose and t%10==0:
                qe = self.quantization_error(intervals)
                print(f"[Batch] Iter {t}, QE={qe:.4f}")
    
    def _update_prototypes_batch(self, data, data_m, data_l, neighbor_w):
        """
        Weighted median update from Chavent & Lechevallier approach:
         for each neuron (i,j), each dimension d:
           m_k^d = WeightedMedian of data_m[:,d], weights= sum( neighbor_w[i,j,s] )
           l_k^d = WeightedMedian of data_l[:,d], same weights
         => [u_k^d, v_k^d] = [m_k^d - l_k^d, m_k^d + l_k^d]
        neighbor_w[i,j,s] => h_{(i,j),c(s)}
        """
        N = data.shape[0]
        for i in range(self.x):
            for j in range(self.y):
                # shape(N,) => for each sample s => w_s
                w_s = neighbor_w[i,j,:]
                w_sum = w_s.sum()
                # skip if w_sum=0 => no update
                if w_sum<1e-15:
                    continue
                for d in range(self.n_dims):
                    # Weighted median of data_m[:,d]
                    median_m = self._weighted_median(data_m[:,d], w_s)
                    median_l = self._weighted_median(data_l[:,d], w_s)
                    self.prototypes[i,j,d,0] = median_m - median_l
                    self.prototypes[i,j,d,1] = median_m + median_l
                    if self.prototypes[i,j,d,0]>self.prototypes[i,j,d,1]:
                        self.prototypes[i,j,d] = np.sort(self.prototypes[i,j,d])
    
    def _weighted_median(self, values, weights):
        """
        Weighted median => sort by values => find point cumsum >= half
        """
        idx_sort = np.argsort(values)
        vs = values[idx_sort]
        ws = weights[idx_sort]
        wsum = ws.sum()
        half = 0.5*wsum
        cumsum=0
        for i in range(len(vs)):
            cumsum += ws[i]
            if cumsum>=half:
                return vs[i]
        return vs[-1]
    
    def quantization(self, data):
        """
        Return the codebook intervals for each sample => shape(N, n_dims,2).
        """
        dist_mat = self._distance_from_weights(data)
        winners_idx = np.argmin(dist_mat, axis=1)
        coords_i = winners_idx//self.y
        coords_j = winners_idx%self.y
        N = data.shape[0]
        result = np.zeros((N, self.n_dims, 2))
        for s in range(N):
            i,j = coords_i[s], coords_j[s]
            result[s] = self.prototypes[i,j]
        return result

    def quantization_error(self, data):
        """
        average distance to BMU
        """
        dist_mat = self._distance_from_weights(data)
        mins = np.min(dist_mat, axis=1)
        return np.mean(mins)

    def get_prototypes(self):
        """
        return prototypes
        """
        return self.prototypes

    def get_neuron_assignments(self, data, return_indices=False):
        """
        Maps each neuron (i, j) to the samples assigned to it.

        Parameters:
        -----------
        data : array-like, shape (N, n_dims, 2)
            The input data containing N samples, each with n_dims features.
        
        return_indices : bool, optional (default=False)
            - If False, stores the actual sample data in the output dictionary.
            - If True, stores the indices of samples instead.

        Returns:
        --------
        assignment_map : dict
            A dictionary where:
            - If return_indices=False: { (i, j): [sample_1, sample_2, ...] }
            - If return_indices=True:  { (i, j): [index_1, index_2, ...] }
        """
        # Ensure data shape is valid, e.g., (N, n_dims, 2)
        assignment_map = defaultdict(list)

        for i, sample in enumerate(data):
            bmu_pos = self.winner(sample)  # Find the Best Matching Unit (BMU), returns (row, col)
            
            if return_indices:
                assignment_map[bmu_pos].append(i)  # Store sample index
            else:
                assignment_map[bmu_pos].append(sample)  # Store actual sample data

        return assignment_map
    
    def neuron_label_map(self, data, labels, return_mode='dominant'):
        """
        Maps each neuron (i, j) to a label summary based on assigned samples.

        Parameters:
        -----------
        data : array-like, shape (N, n_dims, 2)
            The input data containing N samples.
        
        labels : array-like, shape (N,)
            Labels corresponding to each sample in `data`.

        return_mode : str, optional (default='dominant')
            - 'dominant' : Returns the most frequent label assigned to each neuron.
            - 'counter' : Returns a `Counter` object with the count of all labels assigned.

        Returns:
        --------
        neuron_label_map : dict
            A dictionary where:
            - If return_mode='dominant': { (i, j): most_frequent_label }
            - If return_mode='counter': { (i, j): Counter({label_1: count_1, label_2: count_2, ...}) }
        """

        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length.")

        # Dictionary mapping each neuron (i, j) to the list of assigned labels
        label_map = defaultdict(list)

        # Assign each sample's label to its BMU (Best Matching Unit)
        for sample, lab in zip(data, labels):
            bmu = self.winner(sample)  # Get the neuron (i, j) to which the sample is assigned
            label_map[bmu].append(lab)

        # Aggregate labels for each neuron
        output = {}
        for neuron_pos, label_list in label_map.items():
            c = Counter(label_list)  # Count occurrences of each label

            if return_mode == 'dominant':
                # Return the most frequent label
                most_label, _count = c.most_common(1)[0]
                output[neuron_pos] = most_label
            elif return_mode == 'counter':
                # Return the full label distribution
                output[neuron_pos] = c
            else:
                raise ValueError("return_mode must be 'dominant' or 'counter'.")

        return output
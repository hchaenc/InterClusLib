import pandas as pd
import numpy as np
from numpy.random import RandomState
from warnings import warn
from interClusLib.metric import SIMILARITY_FUNCTIONS, DISTANCE_FUNCTIONS

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

        if distance_func in SIMILARITY_FUNCTIONS:
            self.distance_function = SIMILARITY_FUNCTIONS[distance_func]
            self.isSim = True
        elif distance_func in DISTANCE_FUNCTIONS:
            self.distance_function = DISTANCE_FUNCTIONS[distance_func]
            self.isSim = False
        else:
            valid_funcs = ", ".join(list(SIMILARITY_FUNCTIONS.keys()) + list(DISTANCE_FUNCTIONS.keys()))
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
    
    def _convert_centers_to_intervals(self):
        """
        Convert separate lower and upper bounds (centers_a, centers_b) to interval format.
        """
        # Stack centers_a and centers_b along a new last dimension
        # Shape: (n_clusters, n_dims, 2)
        return np.stack([self.centers_a, self.centers_b], axis=2)

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
        self.centroids_ = self._convert_centers_to_intervals()
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

    def get_crisp_assignments(self):
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
        self.crisp_label = crisp_labels
        return crisp_labels
    
    def compute_metrics_for_k_range(self, intervals, min_clusters=2, max_clusters=10, 
                           metrics=['distortion'], distance_func=None, 
                           m=None, max_iter=None, tol=None, 
                           adaptive_weights=None, random_state=None, 
                           n_init=1):
        """
        Compute evaluation metrics for a range of cluster numbers for fuzzy clustering.
        
        Parameters:
        -----------
        intervals : array-like
            Interval data with shape (n_samples, n_dims, 2)
        min_clusters : int, default=2
            Minimum number of clusters to evaluate
        max_clusters : int, default=10
            Maximum number of clusters to evaluate
        metrics : list of str, default=['distortion']
            Metrics to compute, can be any key from the EVALUATION dictionary
        distance_func : str or callable, default=None
            Distance function name or callable. If None, uses the current instance's distance function.
        m : float, default=None
            Fuzzifier parameter. If None, uses the current instance's value.
        max_iter : int, default=None
            Maximum number of iterations. If None, uses the current instance's value.
        tol : float, default=None
            Convergence tolerance. If None, uses the current instance's value.
        adaptive_weights : bool, default=None
            Whether to use adaptive weights. If None, uses the current instance's value.
        random_state : int, default=None
            Random seed. If provided, it will be used to set numpy's random state.
        n_init : int, default=1
            Number of times to run the algorithm with different initializations.
        
        Returns:
        --------
        dict
            Dictionary where keys are metric names and values are dictionaries 
            mapping k values to metric results
        """
        from interClusLib.evaluation import EVALUATION
        
        # Check if requested metrics are valid
        for metric in metrics:
            if metric not in EVALUATION:
                raise ValueError(f"Unknown metric: {metric}. Available options: {list(EVALUATION.keys())}")
        
        # Use current instance parameters if not specified
        distance_func = distance_func or self.distance_function.__name__ if hasattr(self.distance_function, '__name__') else self.distance_func_name if hasattr(self, 'distance_func_name') else 'euclidean'
        m = m or self.m
        max_iter = max_iter or self.max_iter
        tol = tol or self.tol
        adaptive_weights = adaptive_weights if adaptive_weights is not None else self.adaptive_weights
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize results dictionary
        results = {metric: {} for metric in metrics}
        
        # Compute metrics for each k value
        for k in range(min_clusters, max_clusters + 1):
            best_objective = float('inf')
            best_model = None
            
            # Run multiple initializations
            for init in range(n_init):
                try:
                    # Create a new model instance with the current k
                    model = self.__class__(
                        n_clusters=k,
                        m=m,
                        max_iter=max_iter,
                        tol=tol,
                        adaptive_weights=adaptive_weights,
                        distance_func=distance_func
                    )
                    
                    # Fit the model
                    model.fit(intervals)
                    
                    # Keep track of the best model based on objective function
                    objective = model.get_objective()
                    if objective < best_objective:
                        best_objective = objective
                        best_model = model
                        
                except Exception as e:
                    print(f"Error fitting model with k={k}, initialization {init}: {e}")
            
            if best_model is None:
                print(f"Failed to fit model for k={k}, skipping")
                continue
                
            # Get crisp assignments for evaluation metrics
            labels = best_model.get_crisp_assignments()
            
            # Calculate all requested metrics
            for metric in metrics:
                try:
                    metric_func = EVALUATION[metric]
                    metric_value = metric_func(
                        data=intervals,
                        labels=labels,
                        centers=best_model.centroids_,  # Use pre-computed centroids
                        metric=distance_func
                    )
                    results[metric][k] = metric_value
                except Exception as e:
                    print(f"Error calculating {metric} for k={k}: {e}")
        
        return results
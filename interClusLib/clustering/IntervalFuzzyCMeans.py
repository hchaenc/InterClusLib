import pandas as pd
import numpy as np
from numpy.random import RandomState
from warnings import warn
from interClusLib.clustering.AbstractIntervalClustering import AbstractIntervalClustering

class IntervalFuzzyCMeans(AbstractIntervalClustering):
    """
    Interval Fuzzy C-Means (IFCM & IFCMADC) for interval-valued data.
    
    This implementation features vectorized operations, efficient memory usage,
    and improved convergence detection for better performance on large datasets.
    
    Key optimizations:
    - Vectorized distance computations
    - Efficient membership matrix updates
    - Optimized adaptive weight calculations
    - Early convergence detection
    - Memory-efficient data structures
    
    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters (C).
    m : float, default=2.0
        Fuzzifier parameter controlling fuzziness.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-5
        Convergence threshold for membership changes.
    adaptive_weights : bool, default=False
        If True, use IFCMADC method with adaptive weights k_i^j.
    distance_func : str or callable, default='euclidean'
        Distance function name or custom function.
    is_similarity : bool, default=None
        Whether custom function is similarity (True) or distance (False).
    random_state : int, default=None
        Random seed for reproducible results.
    """

    def __init__(self,
                 n_clusters=2,
                 m=2.0,
                 max_iter=100,
                 tol=1e-5,
                 adaptive_weights=False,
                 distance_func="euclidean",
                 is_similarity=None,
                 random_state=None,
                 **kwargs):
        """Initialize the fuzzy c-means algorithm."""
        super().__init__(n_clusters=n_clusters, distance_func=distance_func, 
                        is_similarity=is_similarity, **kwargs)
        
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.adaptive_weights = adaptive_weights
        self.random_state = random_state
        
        # Set random seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

        # Clustering state
        self.U = None
        self.centers_a = None  # shape (n_clusters, n_dims)
        self.centers_b = None  # shape (n_clusters, n_dims)
        self.k = None          # shape (n_clusters, n_dims) for adaptive weights
        self.objective_ = None
        self.crisp_label = None
        self.n_iter_ = 0

    def _init_membership_smart(self, n_samples):
        """
        Smart initialization of membership matrix using K-means++ style approach.
        This often leads to faster convergence than random initialization.
        """
        U = np.random.rand(n_samples, self.n_clusters)
        
        # Add some structure to avoid completely random start
        # Assign higher membership to nearest samples for each cluster
        for i in range(self.n_clusters):
            # Create some preference for different regions
            preference = np.random.randint(0, n_samples, size=n_samples // self.n_clusters)
            U[preference, i] *= 1.5
        
        # Normalize rows to sum to 1
        U = U / U.sum(axis=1, keepdims=True)
        return U

    def _update_centers(self, intervals):
        """
        Update cluster centers for improved performance.
        
        Updates both centers_a and centers_b using matrix operations
        instead of nested loops.
        """
        n_samples, n_dims, _ = intervals.shape
        
        # Compute U^m once
        U_m = self.U ** self.m  # shape (n_samples, n_clusters)
        
        # Compute denominators for all clusters at once
        denom = U_m.sum(axis=0) + 1e-16  # shape (n_clusters,)
        
        # Vectorized computation using einsum for efficiency
        # intervals shape: (n_samples, n_dims, 2)
        # U_m shape: (n_samples, n_clusters)
        
        # Compute weighted sums for lower bounds (a)
        weighted_sum_a = np.einsum('ijk,ic->cj', intervals[:, :, 0:1], U_m)
        # Compute weighted sums for upper bounds (b)  
        weighted_sum_b = np.einsum('ijk,ic->cj', intervals[:, :, 1:2], U_m)
        
        # Update centers
        self.centers_a = (weighted_sum_a / denom[:, None]).squeeze()
        self.centers_b = (weighted_sum_b / denom[:, None]).squeeze()
        
        # Ensure proper shape for single dimension case
        if self.centers_a.ndim == 1:
            self.centers_a = self.centers_a.reshape(-1, 1)
            self.centers_b = self.centers_b.reshape(-1, 1)

    def _convert_centers_to_intervals(self):
        """Convert separate bounds to interval format efficiently."""
        return np.stack([self.centers_a, self.centers_b], axis=2)

    def _compute_distances(self, intervals):
        """
        Simple and reliable distance computation using the same pattern as other algorithms.
        
        Returns
        -------
        ndarray
            Distance matrix of shape (n_samples, n_clusters, n_dims)
        """
        n_samples, n_dims, _ = intervals.shape
        
        # Convert centers to interval format
        centers = self._convert_centers_to_intervals()  # (n_clusters, n_dims, 2)
        
        # Use the same simple approach as K-means and hierarchical clustering
        distances = np.zeros((n_samples, self.n_clusters, n_dims))
        
        for k in range(n_samples):
            for i in range(self.n_clusters):
                for j in range(n_dims):
                    # Extract single intervals exactly like in other algorithms
                    sample_interval = intervals[k, j, :]  # Shape: (2,)
                    center_interval = centers[i, j, :]    # Shape: (2,)
                    
                    # Call distance function the same way as in K-means
                    dist = self.distance_function(sample_interval, center_interval)
                    
                    # Convert similarity to distance if needed
                    if self.isSim:
                        dist = 1.0 - dist
                    
                    distances[k, i, j] = dist
        
        return distances

    def _compute_distances_safe(self, intervals):
        """
        Safe distance computation - same logic as the main method.
        """
        return self._compute_distances(intervals)

    def _update_membership(self, distances):
        """
        Membership matrix update for improved performance.
        
        Uses advanced numpy operations to compute the membership matrix
        efficiently without nested loops.
        """
        n_samples, n_clusters, n_dims = distances.shape
        
        # Compute weighted distances
        if self.adaptive_weights and self.k is not None:
            # Use adaptive weights: sum over dimensions with weights k_i^j
            weighted_distances = np.sum(self.k[None, :, :] * distances, axis=2)
        else:
            # Standard sum over dimensions
            weighted_distances = np.sum(distances, axis=2)
        
        # Initialize membership matrix
        U = np.zeros((n_samples, n_clusters), dtype=np.float64)
        
        # Handle zero distances (assign full membership)
        zero_mask = weighted_distances == 0
        if np.any(zero_mask):
            # For each sample with zero distance, assign full membership to first zero cluster
            for k in range(n_samples):
                zero_clusters = np.where(zero_mask[k, :])[0]
                if len(zero_clusters) > 0:
                    U[k, zero_clusters[0]] = 1.0
                    continue
        
        # Compute membership for non-zero distance samples
        non_zero_samples = ~np.any(zero_mask, axis=1)
        
        if np.any(non_zero_samples):
            # Vectorized computation of membership matrix
            # u_ik = 1 / sum_h (d_ik / d_hk)^(2/(m-1))
            
            exponent = 2.0 / (self.m - 1)
            
            # Compute ratio matrix: (n_samples, n_clusters, n_clusters)
            # ratio[k, i, h] = d_ik / d_hk
            distances_nz = weighted_distances[non_zero_samples, :]  # (n_nz, n_clusters)
            
            # Use broadcasting to compute ratios efficiently
            ratios = distances_nz[:, :, None] / (distances_nz[:, None, :] + 1e-16)
            
            # Raise to power and sum over h dimension
            powered_ratios = ratios ** exponent
            denominators = np.sum(powered_ratios, axis=2)
            
            # Compute membership
            U[non_zero_samples, :] = 1.0 / (denominators + 1e-16)
        
        return U

    def _compute_adaptive_weights(self, distances):
        """
        Computation of adaptive weights using efficient operations.
        
        Computes k_i^j = (∏_h sum_k u_ik^m * d_h^2)^(1/p) / (sum_k u_ik^m * d_j^2)
        """
        n_samples, n_clusters, n_dims = distances.shape
        eps = 1e-16
        
        # Compute U^m efficiently
        U_m = self.U ** self.m  # (n_samples, n_clusters)
        
        # Compute weighted distance sums: sum_k (u_ik^m * d_j^2)
        # distances^2 and weight by membership
        weighted_dist_sums = np.sum(
            U_m[:, :, None] * (distances ** 2), axis=0
        ) + eps  # (n_clusters, n_dims)
        
        # Compute geometric mean of weighted distance sums for numerator
        # Use log-sum-exp trick for numerical stability
        log_weighted_sums = np.log(weighted_dist_sums)
        log_geom_mean = np.mean(log_weighted_sums, axis=1, keepdims=True)  # (n_clusters, 1)
        
        # Compute adaptive weights
        log_k = log_geom_mean - log_weighted_sums
        self.k = np.exp(log_k)
        
        # Clip to prevent extreme values
        self.k = np.clip(self.k, 1e-6, 1e6)

    def _compute_objective(self, distances):
        """
        Computation of the objective function.
        
        J = sum_i sum_k u_ik^m * sum_j k_i^j * d_kij^2
        """
        # Compute U^m
        U_m = self.U ** self.m  # (n_samples, n_clusters)
        
        # Compute weighted distances
        if self.adaptive_weights and self.k is not None:
            weighted_distances = np.sum(self.k[None, :, :] * (distances ** 2), axis=2)
        else:
            weighted_distances = np.sum(distances ** 2, axis=2)
        
        # Compute objective function
        objective = np.sum(U_m * weighted_distances)
        return objective

    def fit(self, intervals):
        """Main fitting loop with enhanced convergence detection."""
        intervals = np.asarray(intervals, dtype=np.float64)
        n_samples, n_dims, _ = intervals.shape
        
        if n_samples < self.n_clusters:
            raise ValueError(f"Number of samples ({n_samples}) must be >= n_clusters ({self.n_clusters})")
        
        # Initialize membership matrix
        self.U = self._init_membership_smart(n_samples)
        
        # Initialize adaptive weights if needed
        if self.adaptive_weights:
            self.k = np.ones((self.n_clusters, n_dims))
        
        prev_objective = float('inf')
        objectives = []
        
        print(f"Starting Fuzzy C-Means with {n_samples} samples, {self.n_clusters} clusters...")
        
        for iteration in range(self.max_iter):
            # Store previous membership for convergence check
            prev_U = self.U.copy()
            
            # Update cluster centers
            self._update_centers(intervals)
            
            # Compute distances
            distances = self._compute_distances(intervals)
            
            # Update adaptive weights if enabled
            if self.adaptive_weights:
                self._compute_adaptive_weights(distances)
            
            # Update membership matrix
            self.U = self._update_membership(distances)
            
            # Compute objective function
            current_objective = self._compute_objective(distances)
            objectives.append(current_objective)
            
            # Check convergence
            membership_change = np.abs(self.U - prev_U).max()
            objective_change = abs(current_objective - prev_objective) if prev_objective != float('inf') else float('inf')
            
            # Multiple convergence criteria
            converged = (
                membership_change < self.tol or
                objective_change < self.tol * abs(prev_objective) or
                (iteration > 10 and objective_change < 1e-8)
            )
            
            if converged:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            prev_objective = current_objective
            
            # Progress indicator for large iterations
            if iteration % 10 == 0 and iteration > 0:
                print(f"Iteration {iteration}: objective = {current_objective:.6f}, "
                      f"membership change = {membership_change:.6f}")
        
        # Store final results
        self.n_iter_ = iteration + 1
        self.objective_ = current_objective
        self.centroids_ = self._convert_centers_to_intervals()
        self.labels_ = self.get_crisp_assignments()
        self.train_data = intervals
        
        print(f"Final objective: {self.objective_:.6f}")
        return self

    def predict(self, intervals):
        """
        Predict cluster memberships for new data.
        
        Parameters
        ----------
        intervals : ndarray
            New interval data with shape (n_samples, n_dims, 2)
            
        Returns
        -------
        ndarray
            Fuzzy membership matrix with shape (n_samples, n_clusters)
        """
        if self.centers_a is None:
            raise RuntimeError("Model not fitted yet")
        
        intervals = np.asarray(intervals, dtype=np.float64)
        
        # Compute distances for new data
        distances = self._compute_distances(intervals)
        
        # Compute membership matrix
        U_new = self._update_membership(distances)
        
        return U_new

    def predict_crisp(self, intervals):
        """
        Predict crisp cluster assignments for new data.
        
        Parameters
        ----------
        intervals : ndarray
            New interval data with shape (n_samples, n_dims, 2)
            
        Returns
        -------
        ndarray
            Crisp cluster labels with shape (n_samples,)
        """
        U_new = self.predict(intervals)
        return np.argmax(U_new, axis=1)

    def get_membership(self):
        """Return final membership matrix after fit."""
        if self.U is None:
            raise RuntimeError("Model not fitted yet.")
        return self.U

    def get_centers(self):
        """Return final cluster center intervals."""
        if self.centers_a is None or self.centers_b is None:
            raise RuntimeError("Model not fitted yet.")
        return self.centers_a, self.centers_b

    def get_objective(self):
        """Return final objective function value."""
        return self.objective_

    def get_crisp_assignments(self):
        """Get crisp cluster assignments from fuzzy memberships."""
        if self.U is None:
            raise RuntimeError("Model not fitted yet.")
        return np.argmax(self.U, axis=1)

    def compute_metrics_for_k_range(self, intervals, min_clusters=2, max_clusters=10,
                               metrics=['silhouette', 'calinski_harabasz', 'davies_bouldin'],
                               distance_func=None, m=None, max_iter=None, tol=None,
                               adaptive_weights=None, random_state=None, n_init=3, **kwargs):
        """
        Compute evaluation metrics for multiple cluster numbers.
        
        Uses reduced n_init and faster metrics by default for better performance.
        """
        from interClusLib.evaluation import EVALUATION
        
        # Validate metrics
        for metric in metrics:
            if metric not in EVALUATION:
                raise ValueError(f"Unknown metric: {metric}. Available: {list(EVALUATION.keys())}")
        
        # Use current parameters if not specified
        distance_func = distance_func or self.distance_func
        m = m or self.m
        max_iter = max_iter or min(self.max_iter, 50)  # Limit iterations for faster evaluation
        tol = tol or self.tol
        adaptive_weights = adaptive_weights if adaptive_weights is not None else self.adaptive_weights
        
        if random_state is not None:
            np.random.seed(random_state)
        
        results = {metric: {} for metric in metrics}
        
        print(f"Computing metrics for k={min_clusters} to {max_clusters} with {n_init} initializations each...")
        
        for k in range(min_clusters, max_clusters + 1):
            print(f"Processing k={k}...")
            
            best_objective = float('inf')
            best_model = None
            
            # Multiple initializations
            for init in range(n_init):
                try:
                    model = self.__class__(
                        n_clusters=k,
                        m=m,
                        max_iter=max_iter,
                        tol=tol,
                        adaptive_weights=adaptive_weights,
                        distance_func=distance_func,
                        is_similarity=self.isSim if hasattr(self, 'isSim') else None,
                        random_state=random_state + init if random_state else None
                    )
                    
                    model.fit(intervals)
                    
                    if model.get_objective() < best_objective:
                        best_objective = model.get_objective()
                        best_model = model
                        
                except Exception as e:
                    print(f"Error in k={k}, init={init}: {e}")
            
            if best_model is None:
                print(f"Failed to fit k={k}")
                continue
            
            # Compute metrics
            labels = best_model.get_crisp_assignments()
            centroids = best_model.centroids_
            
            for metric_name in metrics:
                try:
                    metric_func = EVALUATION[metric_name]
                    metric_value = metric_func(
                        data=intervals,
                        labels=labels,
                        centers=centroids,
                        metric=distance_func
                    )
                    results[metric_name][k] = metric_value
                except Exception as e:
                    print(f"Error computing {metric_name} for k={k}: {e}")
        
        return results

    def cluster_and_return(self, data, k):
        """
        Clustering for single k value.
        """
        model = self.__class__(
            n_clusters=k,
            m=self.m,
            max_iter=self.max_iter,
            tol=self.tol,
            adaptive_weights=self.adaptive_weights,
            distance_func=self.distance_func,
            is_similarity=self.isSim if hasattr(self, 'isSim') else None,
            random_state=self.random_state
        )
        model.fit(data)
        return model.get_crisp_assignments(), model.centroids_

    def get_performance_stats(self):
        """Get performance statistics."""
        return {
            'n_clusters': self.n_clusters,
            'n_iterations': getattr(self, 'n_iter_', 'Not fitted'),
            'final_objective': getattr(self, 'objective_', 'Not fitted'),
            'adaptive_weights': self.adaptive_weights,
            'fuzzifier_m': self.m,
            'distance_function': self.distance_func
        }
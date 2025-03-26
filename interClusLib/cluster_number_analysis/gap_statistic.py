"""
Gap Statistic implementation module
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings
from .base_evaluator import ClusterEvaluationMethod

class GapStatistic(ClusterEvaluationMethod):
    """Gap Statistic implementation class"""
    
    def __init__(self, min_clusters=2, max_clusters=20, n_refs=10, random_state=None):
        """
        Initialize the Gap Statistic.
        
        Parameters:
        min_clusters: int, minimum number of clusters to consider
        max_clusters: int, maximum number of clusters to consider
        n_refs: int, number of reference datasets
        random_state: int, random seed
        """
        super().__init__(min_clusters, max_clusters)
        self.n_refs = n_refs
        self.random_state = random_state
        self.gap_stats = None
        self.ref_dispersions = None
    
    def evaluate(self, eval_data, raw_data=None, cluster_func=None):
        """
        Use the Gap Statistic to determine the optimal number of clusters.
        
        Parameters:
        eval_data: evaluation data containing metrics for different cluster numbers
        raw_data: original data, if provided will be used to generate reference data
        cluster_func: clustering function that accepts data and k, returns labels and centers
        
        Returns:
        int: optimal number of clusters
        """
        data = self._validate_and_format_data(eval_data)

        self.eval_results = data
        
        # Extract k values and corresponding dispersions (assuming lower is better)
        k_values = data[:, 0].astype(int)
        dispersions = data[:, 1]
        log_dispersions = np.log(dispersions)
        
        if raw_data is not None and cluster_func is not None:
            # If original data and clustering function are provided, use complete Gap Statistic calculation
            self.gap_stats, self.ref_dispersions = self._compute_gap_statistic(
                raw_data, k_values, cluster_func)
        else:
            # Otherwise use simplified version based only on provided evaluation data
            warnings.warn("Original data or clustering function not provided, using simplified Gap Statistic calculation")
            # Generate reference model
            if self.random_state is not None:
                np.random.seed(self.random_state)
                
            # For simplification, we only simulate reference model dispersions
            # In real applications, random reference data should be generated and clustered
            ref_log_dispersions = np.zeros((self.n_refs, len(k_values)))
            for i in range(self.n_refs):
                # Simulate reference model log dispersions
                ref_log_dispersions[i] = np.log(np.linspace(dispersions.max(), 
                                                          dispersions.min() * 0.8, 
                                                          len(k_values)))
                ref_log_dispersions[i] += np.random.normal(0, 0.1, len(k_values))
            
            self.ref_dispersions = ref_log_dispersions
            # Calculate Gap Statistic
            self.gap_stats = np.mean(ref_log_dispersions, axis=0) - log_dispersions
        
        # Calculate standard deviation and error
        sdk = np.std(self.ref_dispersions, axis=0) * np.sqrt(1 + 1/self.n_refs)
        
        # Find the smallest k that satisfies Gap(k) ≥ Gap(k+1) - sk+1
        for i in range(len(k_values)-1):
            if self.gap_stats[i] >= self.gap_stats[i+1] - sdk[i+1]:
                self.optimal_k = k_values[i]
                break
        else:
            self.optimal_k = k_values[-1]
        
        self.eval_results = data
        return self.optimal_k
    
    def _compute_gap_statistic(self, data, k_values, cluster_func):
        """
        Compute Gap Statistic
        
        Parameters:
        data: original data
        k_values: cluster numbers to evaluate
        cluster_func: clustering function that accepts data and k, returns labels and centers
        
        Returns:
        tuple: (gap_stats, ref_log_dispersions)
        """
        # Generate reference data
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        
        # Store reference data log dispersions
        ref_log_dispersions = np.zeros((self.n_refs, len(k_values)))
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Calculate reference data dispersions
        for i in range(self.n_refs):
            # Generate uniformly distributed reference data
            ref_data = np.random.uniform(min_vals, max_vals, data.shape)
            
            for j, k in enumerate(k_values):
                labels, centers = cluster_func(ref_data, k)
                
                # Calculate dispersion
                dispersion = 0
                for c in range(k):
                    cluster_points = ref_data[labels == c]
                    if len(cluster_points) > 0:
                        # Calculate within-cluster sum of squares
                        centroid = centers[c]
                        dispersion += np.sum(np.sum((cluster_points - centroid) ** 2, axis=1))
                
                ref_log_dispersions[i, j] = np.log(dispersion) if dispersion > 0 else 0
        
        # Calculate actual data dispersions (already given in eval_data)
        log_dispersions = np.log(self.eval_results[:, 1])
        
        # Calculate Gap Statistic
        gap_stats = np.mean(ref_log_dispersions, axis=0) - log_dispersions
        
        return gap_stats, ref_log_dispersions
    
    def plot_gap(self, figsize=(10, 6)):
        """
        Plot Gap Statistic graph
        
        Parameters:
        figsize: tuple, figure size
        
        Returns:
        plt: matplotlib figure object
        """
        if self.gap_stats is None:
            raise ValueError("Must run evaluate method first")
        
        plt.figure(figsize=figsize)
        
        k_values = self.eval_results[:, 0].astype(int)
        
        # Calculate standard error
        sdk = np.std(self.ref_dispersions, axis=0) * np.sqrt(1 + 1/self.n_refs)
        
        plt.errorbar(k_values, self.gap_stats, yerr=sdk, fmt='o-', capsize=5)
        plt.axvline(x=self.optimal_k, color='red', linestyle='--', 
                  label=f'Optimal k={self.optimal_k}')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Gap Statistic')
        plt.title('Gap Statistic Method')
        plt.legend()
        
        return plt
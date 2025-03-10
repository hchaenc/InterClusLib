"""
L Method implementation module
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from .base_evaluator import ClusterEvaluationMethod

class LMethod(ClusterEvaluationMethod):
    """L Method implementation class"""
    
    def __init__(self, min_clusters=2, max_clusters=20, iterative=True, min_points_per_line=2):
        """
        Initialize the L Method.
        
        Parameters:
        min_clusters: int, minimum number of clusters to consider
        max_clusters: int, maximum number of clusters to consider
        iterative: bool, whether to use the iterative L Method, default is True
        min_points_per_line: int, minimum number of points per line
        """
        super().__init__(min_clusters, max_clusters)
        self.iterative = iterative
        self.min_points_per_line = min_points_per_line
    
    def evaluate(self, eval_data):
        """
        Use the L Method to determine the optimal number of clusters.
        
        Parameters:
        eval_data: dict or array, containing cluster counts and corresponding evaluation metrics
        
        Returns:
        int: optimal number of clusters
        """
        data = self._validate_and_format_data(eval_data)
        
        if self.iterative:
            self.optimal_k = self._iterative_l_method(data)
        else:
            self.optimal_k = self._standard_l_method(data)
        
        self.eval_results = data
        return self.optimal_k
    
    def _standard_l_method(self, data):
        """
        Standard L Method implementation.
        
        Parameters:
        data: sorted evaluation data with shape (n, 2)
        
        Returns:
        int: optimal number of clusters
        """
        n = len(data)
        min_rmse = float('inf')
        knee_idx = 0
        
        # Iterate through all possible knee positions
        for c in range(1, n-1):
            # Ensure each line has enough points
            if c < self.min_points_per_line - 1 or n - c - 1 < self.min_points_per_line - 1:
                continue
                
            # Left sequence Lc
            Lc = data[:c+1]
            # Right sequence Rc
            Rc = data[c+1:]
            
            # Left linear regression
            X_left = Lc[:, 0].reshape(-1, 1)
            y_left = Lc[:, 1]
            model_left = LinearRegression().fit(X_left, y_left)
            y_pred_left = model_left.predict(X_left)
            rmse_left = np.sqrt(np.mean((y_left - y_pred_left) ** 2))
            
            # Right linear regression
            X_right = Rc[:, 0].reshape(-1, 1)
            y_right = Rc[:, 1]
            model_right = LinearRegression().fit(X_right, y_right)
            y_pred_right = model_right.predict(X_right)
            rmse_right = np.sqrt(np.mean((y_right - y_pred_right) ** 2))
            
            # Calculate weighted total RMSE
            w_left = len(Lc) / float(n)
            w_right = len(Rc) / float(n)
            total_rmse = w_left * rmse_left + w_right * rmse_right
            
            if total_rmse < min_rmse:
                min_rmse = total_rmse
                knee_idx = c
        
        # Return the cluster number corresponding to the knee point
        return int(data[knee_idx, 0])
    
    def _iterative_l_method(self, data):
        """
        Iterative improved L Method implementation.
        
        Parameters:
        data: sorted evaluation data with shape (n, 2)
        
        Returns:
        int: optimal number of clusters
        """
        # Initialize
        last_knee = float('inf')
        current_knee = float('inf')
        cutoff = None
        
        # Iterate until convergence
        while True:
            # If no cutoff is specified, use all data
            if cutoff is None:
                focus_data = data
            else:
                # Focus only on data points within cutoff
                focus_data = data[data[:, 0] <= cutoff]
                
                # Ensure we have enough data points
                if len(focus_data) < max(5, 2 * self.min_points_per_line):
                    focus_data = data[:min(max(5, 2 * self.min_points_per_line), len(data))]
            
            # Use standard L Method to find the knee in the current focus area
            knee_idx = 0
            min_rmse = float('inf')
            n = len(focus_data)
            
            for c in range(1, n-1):
                # Ensure each line has enough points
                if c < self.min_points_per_line - 1 or n - c - 1 < self.min_points_per_line - 1:
                    continue
                    
                Lc = focus_data[:c+1]
                Rc = focus_data[c+1:]
                
                X_left = Lc[:, 0].reshape(-1, 1)
                y_left = Lc[:, 1]
                model_left = LinearRegression().fit(X_left, y_left)
                y_pred_left = model_left.predict(X_left)
                rmse_left = np.sqrt(np.mean((y_left - y_pred_left) ** 2))
                
                X_right = Rc[:, 0].reshape(-1, 1)
                y_right = Rc[:, 1]
                model_right = LinearRegression().fit(X_right, y_right)
                y_pred_right = model_right.predict(X_right)
                rmse_right = np.sqrt(np.mean((y_right - y_pred_right) ** 2))
                
                w_left = len(Lc) / float(n)
                w_right = len(Rc) / float(n)
                total_rmse = w_left * rmse_left + w_right * rmse_right
                
                if total_rmse < min_rmse:
                    min_rmse = total_rmse
                    knee_idx = c
            
            current_knee = int(focus_data[knee_idx, 0])
            
            # If the knee no longer moves left or reaches minimum clusters, converge
            if current_knee >= last_knee or current_knee <= self.min_clusters:
                break
            
            # Update results and set new cutoff (twice the knee point)
            last_knee = current_knee
            cutoff = current_knee * 2
        
        return current_knee
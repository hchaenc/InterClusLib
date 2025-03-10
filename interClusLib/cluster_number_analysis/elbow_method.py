"""
Elbow Method implementation module
"""
import numpy as np
from scipy.signal import savgol_filter
from .base_evaluator import ClusterEvaluationMethod

class ElbowMethod(ClusterEvaluationMethod):
    """Elbow Method implementation class"""
    
    def __init__(self, min_clusters=2, max_clusters=20, second_derivative=False, smooth=False):
        """
        Initialize the Elbow Method.
        
        Parameters:
        min_clusters: int, minimum number of clusters to consider
        max_clusters: int, maximum number of clusters to consider
        second_derivative: bool, whether to use the second derivative method
        smooth: bool, whether to apply data smoothing
        """
        super().__init__(min_clusters, max_clusters)
        self.second_derivative = second_derivative
        self.smooth = smooth
    
    def evaluate(self, eval_data):
        """
        Use the Elbow Method to determine the optimal number of clusters.
        
        Parameters:
        eval_data: dict or array, containing cluster counts and corresponding evaluation metrics
        
        Returns:
        int: optimal number of clusters
        """
        data = self._validate_and_format_data(eval_data)
        
        x = data[:, 0]
        y = data[:, 1]
        
        # Apply data smoothing (optional)
        if self.smooth and len(y) >= 5:
            window_length = min(5, len(y) - 2)
            if window_length % 2 == 0:  # savgol_filter requires odd window length
                window_length += 1
            y_smooth = savgol_filter(y, window_length=window_length, polyorder=2)
        else:
            y_smooth = y
            
        if self.second_derivative:
            # Use second derivative method
            # Calculate first differences
            diffs = np.diff(y_smooth) / np.diff(x)
            
            # Calculate second differences
            second_diffs = np.diff(diffs)
            
            # Get index of max/min second difference
            if np.mean(y_smooth) >= 0:  # For increasing curves, find convex knee
                idx = np.argmax(second_diffs) + 1
            else:  # For decreasing curves, find concave knee
                idx = np.argmin(second_diffs) + 1
                
            self.optimal_k = int(x[idx])
        else:
            # Use maximum curvature method
            # Calculate curvature
            dx_dt = np.gradient(x)
            dy_dt = np.gradient(y_smooth)
            d2y_dt2 = np.gradient(dy_dt)
            
            # Calculate curvature
            curvature = np.abs(d2y_dt2) / (1 + dy_dt**2)**1.5
            
            # Get index of maximum curvature
            idx = np.argmax(curvature)
            self.optimal_k = int(x[idx])
            
        self.eval_results = data
        return self.optimal_k
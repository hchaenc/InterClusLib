from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from enum import Enum
import warnings
from interClusLib.IntervalData import IntervalData

def impute_missing_intervals(interval_data, missing_mask=None, lower_missing_mask=None, upper_missing_mask=None, method='mean'):
    """Impute missing interval data, including cases where only lower or upper bounds are missing
    
    Methods:
        - 'mean': Use the average midpoint and width
        - 'median': Use the median midpoint and width
        - 'knn': Imputation based on k-nearest neighbors (not implemented)
        - 'remove': Remove samples with any missing values instead of imputing
    
    Parameters:
    -----------
    interval_data: numpy.ndarray
        Interval data with shape (n_samples, n_dim, 2)
    missing_mask: numpy.ndarray, optional
        Boolean mask indicating completely missing intervals with shape (n_samples, n_dim)
    lower_missing_mask: numpy.ndarray, optional
        Boolean mask indicating missing lower bounds with shape (n_samples, n_dim)
    upper_missing_mask: numpy.ndarray, optional
        Boolean mask indicating missing upper bounds with shape (n_samples, n_dim)
    method: str, default='mean'
        Method to use for imputation
        
    Returns:
    --------
    numpy.ndarray
        Imputed interval data with shape (n_samples, n_dim, 2) if method is not 'remove'
        or (n_valid_samples, n_dim, 2) if method is 'remove'
    """
    n_samples, n_dim, _ = interval_data.shape
    imputed_data = interval_data.copy()
    
    # Initialize masks if not provided
    if missing_mask is None:
        missing_mask = np.zeros((n_samples, n_dim), dtype=bool)
    if lower_missing_mask is None:
        lower_missing_mask = np.zeros((n_samples, n_dim), dtype=bool)
    if upper_missing_mask is None:
        upper_missing_mask = np.zeros((n_samples, n_dim), dtype=bool)
    
    # Create a combined mask for any type of missing values
    any_missing_mask = missing_mask | lower_missing_mask | upper_missing_mask
    
    # If using remove method, return only samples without any missing values
    if method == 'remove':
        # Find samples that have at least one missing value
        samples_to_remove = np.any(any_missing_mask, axis=1)
        # Keep only samples without missing values
        return interval_data[~samples_to_remove]
    
    for d in range(n_dim):
        # Handle completely missing intervals
        missing_idx = missing_mask[:, d]
        lower_missing_idx = lower_missing_mask[:, d]
        upper_missing_idx = upper_missing_mask[:, d]
        
        # If no missing data in this dimension, skip
        if not np.any(missing_idx) and not np.any(lower_missing_idx) and not np.any(upper_missing_idx):
            continue
            
        # Get valid data (neither lower nor upper bounds are missing)
        valid_mask = ~(missing_idx | lower_missing_idx | upper_missing_idx)
        valid_data = interval_data[valid_mask, d, :]
        
        if len(valid_data) == 0:
            # If no valid data, skip this dimension or use some default
            continue
        
        if method == 'mean':
            # Calculate average midpoint and width
            midpoints = (valid_data[:, 0] + valid_data[:, 1]) / 2
            widths = valid_data[:, 1] - valid_data[:, 0]
            avg_midpoint = np.mean(midpoints)
            avg_width = np.mean(widths)
            
            # Impute completely missing intervals
            if np.any(missing_idx):
                imputed_data[missing_idx, d, 0] = avg_midpoint - avg_width/2
                imputed_data[missing_idx, d, 1] = avg_midpoint + avg_width/2
            
            # Impute missing lower bounds
            if np.any(lower_missing_idx):
                upper_values = imputed_data[lower_missing_idx, d, 1]
                imputed_data[lower_missing_idx, d, 0] = upper_values - avg_width
            
            # Impute missing upper bounds
            if np.any(upper_missing_idx):
                lower_values = imputed_data[upper_missing_idx, d, 0]
                imputed_data[upper_missing_idx, d, 1] = lower_values + avg_width
                
        elif method == 'median':
            # Calculate median midpoint and width
            midpoints = (valid_data[:, 0] + valid_data[:, 1]) / 2
            widths = valid_data[:, 1] - valid_data[:, 0]
            med_midpoint = np.median(midpoints)
            med_width = np.median(widths)
            
            # Impute completely missing intervals
            if np.any(missing_idx):
                imputed_data[missing_idx, d, 0] = med_midpoint - med_width/2
                imputed_data[missing_idx, d, 1] = med_midpoint + med_width/2
            
            # Impute missing lower bounds
            if np.any(lower_missing_idx):
                upper_values = imputed_data[lower_missing_idx, d, 1]
                imputed_data[lower_missing_idx, d, 0] = upper_values - med_width
            
            # Impute missing upper bounds
            if np.any(upper_missing_idx):
                lower_values = imputed_data[upper_missing_idx, d, 0]
                imputed_data[upper_missing_idx, d, 1] = lower_values + med_width
    
    return imputed_data


class MissingValueAction(Enum):
    """Actions to take when missing values are found"""
    IGNORE = "ignore"           # Keep missing values as-is
    DROP_ROWS = "drop_rows"     # Remove rows with missing values
    FILL_MEAN = "fill_mean"     # Fill with mean of valid values
    FILL_MEDIAN = "fill_median" # Fill with median of valid values
    FILL_MODE = "fill_mode"     # Fill with most frequent value
    FILL_FORWARD = "fill_forward" # Forward fill
    FILL_BACKWARD = "fill_backward" # Backward fill
    FILL_CUSTOM = "fill_custom" # Fill with custom value


class MissingValueImputor:
    """
    Handler for detecting and processing missing values in interval data
    
    Missing values in interval data can occur in lower bounds, upper bounds, or both.
    This handler provides various strategies for detecting and handling such missing values.
    """
    
    def __init__(self):
        """
        Initialize the missing value handler
        """
        self._last_check_results = None
    
    def check_missing_values(self, interval_data) -> Dict[str, any]:
        """
        Check for missing values in the interval data
        
        Args:
            interval_data: IntervalData instance to check
            
        Returns:
            Dictionary containing check results with keys:
            - 'has_missing': bool, whether any missing values exist
            - 'total_missing': int, total number of missing intervals
            - 'missing_by_feature': dict, missing count per feature
            - 'missing_rows': list, row indices with missing values
            - 'missing_details': list of dicts with detailed information
        """
        results = {
            'has_missing': False,
            'total_missing': 0,
            'missing_by_feature': {},
            'missing_rows': set(),
            'missing_details': []
        }
        
        # Check each interval feature
        for feature in interval_data.features:
            feature_results = self._check_feature_missing_values(interval_data, feature)
            
            if feature_results['count'] > 0:
                results['has_missing'] = True
                results['total_missing'] += feature_results['count']
                results['missing_by_feature'][feature] = feature_results['count']
                results['missing_rows'].update(feature_results['rows'])
                results['missing_details'].extend(feature_results['details'])
        
        # Convert set to sorted list for consistency
        results['missing_rows'] = sorted(list(results['missing_rows']))
        
        # Cache results
        self._last_check_results = results
        return results
    
    def _check_feature_missing_values(self, interval_data, feature: str) -> Dict[str, any]:
        """Check missing values for a specific feature"""
        # Find the column pair for this feature
        lower_col = None
        upper_col = None
        
        for lower, upper in interval_data.interval_pairs:
            if feature in lower and feature in upper:
                lower_col = lower
                upper_col = upper
                break
        
        if lower_col is None or upper_col is None:
            return {'count': 0, 'rows': [], 'details': []}
        
        # Get the data
        lower_values = interval_data.data[lower_col]
        upper_values = interval_data.data[upper_col]
        
        # Find missing values
        lower_missing = pd.isna(lower_values)
        upper_missing = pd.isna(upper_values)
        any_missing = lower_missing | upper_missing
        
        missing_indices = np.where(any_missing)[0]
        
        # Create detailed information
        details = []
        for idx in missing_indices:
            lower_is_missing = pd.isna(lower_values.iloc[idx])
            upper_is_missing = pd.isna(upper_values.iloc[idx])
            
            missing_type = 'both' if (lower_is_missing and upper_is_missing) else \
                          'lower' if lower_is_missing else 'upper'
            
            details.append({
                'row': idx,
                'feature': feature,
                'lower_column': lower_col,
                'upper_column': upper_col,
                'lower_value': lower_values.iloc[idx],
                'upper_value': upper_values.iloc[idx],
                'lower_missing': lower_is_missing,
                'upper_missing': upper_is_missing,
                'missing_type': missing_type
            })
        
        return {
            'count': len(missing_indices),
            'rows': missing_indices.tolist(),
            'details': details
        }
    
    def fix_missing_values(self, interval_data, 
                          action: Union[str, MissingValueAction] = MissingValueAction.DROP_ROWS,
                          fill_value: Optional[float] = None,
                          inplace: bool = False):
        """
        Fix missing values using the specified action
        
        Args:
            interval_data: IntervalData instance to fix
            action: Action to take (drop_rows, fill_mean, fill_median, etc.)
            fill_value: Custom value to fill with (for FILL_CUSTOM action)
            inplace: Whether to modify the original data or return a copy
            
        Returns:
            IntervalData instance with fixed missing values
        """
        if isinstance(action, str):
            try:
                action = MissingValueAction(action.lower())
            except ValueError:
                raise ValueError(f"Invalid action: {action}. Must be one of: {[e.value for e in MissingValueAction]}")
        
        # Work on copy unless inplace=True
        if inplace:
            result_data = interval_data
        else:
            result_data = interval_data.copy()
        
        # Check for missing values
        check_results = self.check_missing_values(interval_data)
        
        if not check_results['has_missing']:
            print("No missing values found.")
            return result_data
        
        if action == MissingValueAction.IGNORE:
            print(f"Ignoring {check_results['total_missing']} missing values.")
            return result_data
        
        # Apply the specified action
        if action == MissingValueAction.DROP_ROWS:
            result_data = self._drop_rows_with_missing(result_data, check_results)
        elif action == MissingValueAction.FILL_MEAN:
            result_data = self._fill_with_statistic(result_data, 'mean')
        elif action == MissingValueAction.FILL_MEDIAN:
            result_data = self._fill_with_statistic(result_data, 'median')
        elif action == MissingValueAction.FILL_MODE:
            result_data = self._fill_with_statistic(result_data, 'mode')
        elif action == MissingValueAction.FILL_FORWARD:
            result_data = self._fill_forward_backward(result_data, method='forward')
        elif action == MissingValueAction.FILL_BACKWARD:
            result_data = self._fill_forward_backward(result_data, method='backward')
        elif action == MissingValueAction.FILL_CUSTOM:
            if fill_value is None:
                raise ValueError("fill_value must be provided for FILL_CUSTOM action")
            result_data = self._fill_with_custom_value(result_data, fill_value)
        
        return result_data
    
    def _drop_rows_with_missing(self, interval_data, check_results: Dict):
        """Remove rows containing missing values"""
        data_copy = interval_data.data.copy()
        rows_to_remove = check_results['missing_rows']
        
        if rows_to_remove:
            data_copy = data_copy.drop(index=rows_to_remove).reset_index(drop=True)
            print(f"Removed {len(rows_to_remove)} rows with missing values.")
        
        from __main__ import IntervalData
        return IntervalData(data_copy)
    
    def _fill_with_statistic(self, interval_data, statistic: str):
        """Fill missing values with statistical measures (mean, median, mode)"""
        data_copy = interval_data.data.copy()
        filled_count = 0
        
        for lower_col, upper_col in interval_data.interval_pairs:
            for col in [lower_col, upper_col]:
                if data_copy[col].isna().any():
                    if statistic == 'mean':
                        fill_value = data_copy[col].mean()
                    elif statistic == 'median':
                        fill_value = data_copy[col].median()
                    elif statistic == 'mode':
                        mode_result = data_copy[col].mode()
                        fill_value = mode_result.iloc[0] if len(mode_result) > 0 else data_copy[col].mean()
                    
                    if not pd.isna(fill_value):
                        missing_count = data_copy[col].isna().sum()
                        data_copy = data_copy.fillna({col: fill_value})
                        filled_count += missing_count
        
        print(f"Filled {filled_count} missing values with {statistic}.")
        
        return IntervalData(data_copy)
    
    def _fill_forward_backward(self, interval_data, method: str):
        """Fill missing values using forward or backward fill"""
        data_copy = interval_data.data.copy()
        filled_count = 0
        
        for lower_col, upper_col in interval_data.interval_pairs:
            for col in [lower_col, upper_col]:
                if data_copy[col].isna().any():
                    missing_count = data_copy[col].isna().sum()
                    if method == 'forward':
                        data_copy[col].fillna(method='ffill', inplace=True)
                    else:  # backward
                        data_copy[col].fillna(method='bfill', inplace=True)
                    
                    filled_after = data_copy[col].isna().sum()
                    filled_count += missing_count - filled_after
        
        print(f"Filled {filled_count} missing values using {method} fill.")
        
        return IntervalData(data_copy)
    
    def _fill_with_custom_value(self, interval_data, fill_value: float):
        """Fill missing values with a custom value"""
        data_copy = interval_data.data.copy()
        filled_count = 0
        
        for lower_col, upper_col in interval_data.interval_pairs:
            for col in [lower_col, upper_col]:
                if data_copy[col].isna().any():
                    missing_count = data_copy[col].isna().sum()
                    data_copy[col].fillna(fill_value, inplace=True)
                    filled_count += missing_count
        
        print(f"Filled {filled_count} missing values with custom value {fill_value}.")
        
        return IntervalData(data_copy)
    
    def print_report(self, interval_data, detailed: bool = False) -> None:
        """
        Print a simple report about missing values
        
        Args:
            interval_data: IntervalData instance to analyze
            detailed: Whether to show detailed information for each missing value
        """
        results = self.check_missing_values(interval_data)
        
        print("=== Missing Value Analysis Report ===")
        print(f"Total samples: {len(interval_data)}")
        print(f"Total features: {interval_data.n_features}")
        print(f"Has missing values: {results['has_missing']}")
        print(f"Total missing intervals: {results['total_missing']}")
        
        if results['has_missing']:
            print(f"Affected rows: {len(results['missing_rows'])}")
            print(f"Percentage of affected samples: {len(results['missing_rows'])/len(interval_data)*100:.2f}%")
            
            print("\nMissing values by feature:")
            for feature, count in results['missing_by_feature'].items():
                percentage = count / len(interval_data) * 100
                print(f"  {feature}: {count} ({percentage:.2f}%)")
            
            if detailed and results['missing_details']:
                print("\nDetailed missing value information:")
                for i, detail in enumerate(results['missing_details'][:10], 1):  # Show first 10
                    missing_info = []
                    if detail['lower_missing']:
                        missing_info.append("lower")
                    if detail['upper_missing']:
                        missing_info.append("upper")
                    
                    print(f"  {i:2d}. Row {detail['row']:3d}, Feature '{detail['feature']}': "
                          f"Missing {', '.join(missing_info)} bound(s)")
                
                if len(results['missing_details']) > 10:
                    print(f"  ... and {len(results['missing_details']) - 10} more")
        else:
            print("✓ No missing values detected!")
    
    def get_last_results(self) -> Optional[Dict]:
        """Get results from the last check_missing_values call"""
        return self._last_check_results
    
    @staticmethod
    def validate_and_fix(interval_data, 
                        action: Union[str, MissingValueAction] = MissingValueAction.DROP_ROWS,
                        fill_value: Optional[float] = None,
                        show_report: bool = True):
        """
        Convenience method to check and fix missing values in one call
        
        Args:
            interval_data: IntervalData instance to process
            action: Action to take for missing values
            fill_value: Custom fill value (for FILL_CUSTOM action)
            show_report: Whether to print a report
            
        Returns:
            IntervalData instance with fixed missing values
        """
        imputor = MissingValueImputor()
        
        if show_report:
            imputor.print_report(interval_data)
        
        return imputor.fix_missing_values(interval_data, action=action, fill_value=fill_value)
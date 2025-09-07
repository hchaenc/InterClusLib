import pandas as pd
import numpy as np
from interClusLib.IntervalData import IntervalData
from typing import List, Tuple, Dict, Optional, Union
from enum import Enum
import warnings

class InversionAction(Enum):
    """Actions to take when inverted intervals are found"""
    IGNORE = "ignore"           # Keep inverted intervals as-is
    SWAP = "swap"              # Swap lower and upper bounds
    REMOVE = "remove"          # Remove rows with inverted intervals
    SET_NAN = "set_nan"        # Set inverted intervals to NaN


class InvertedIntervalHandler:
    """
    Class for detecting and handling inverted interval data
    
    Inverted intervals occur when the lower bound is greater than the upper bound,
    which violates the fundamental property of intervals.
    """
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize the inverted interval checker
        
        Args:
            tolerance: Small tolerance for numerical comparison (lower > upper + tolerance)
        """
        self.tolerance = tolerance
        self._last_check_results = None
    
    def check_intervals(self, interval_data: 'IntervalData') -> Dict[str, any]:
        """
        Check for inverted intervals in the data (skips rows with missing values)
        
        Args:
            interval_data: IntervalData instance to check
            
        Returns:
            Dictionary containing check results with keys:
            - 'has_inverted': bool, whether any inverted intervals exist
            - 'total_inverted': int, total number of inverted intervals
            - 'inverted_by_feature': dict, inverted count per feature
            - 'inverted_rows': list, row indices with inverted intervals
            - 'inverted_details': list of dicts with detailed information
        """
        results = {
            'has_inverted': False,
            'total_inverted': 0,
            'inverted_by_feature': {},
            'inverted_rows': set(),
            'inverted_details': []
        }
        
        # Check each interval feature
        for feature in interval_data.features:
            feature_results = self._check_feature_intervals(interval_data, feature)
            
            if feature_results['count'] > 0:
                results['has_inverted'] = True
                results['total_inverted'] += feature_results['count']
                results['inverted_by_feature'][feature] = feature_results['count']
                results['inverted_rows'].update(feature_results['rows'])
                results['inverted_details'].extend(feature_results['details'])
        
        # Convert set to sorted list for consistency
        results['inverted_rows'] = sorted(list(results['inverted_rows']))
        
        # Cache results
        self._last_check_results = results
        return results
    
    def _check_feature_intervals(self, interval_data: 'IntervalData', 
                               feature: str) -> Dict[str, any]:
        """Check intervals for a specific feature, skipping rows with missing values"""
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
        
        # Skip rows with missing values
        valid_mask = pd.notna(lower_values) & pd.notna(upper_values)
        
        if not valid_mask.any():
            # No valid intervals to check
            return {'count': 0, 'rows': [], 'details': []}
        
        # Among valid intervals, find inverted ones
        valid_lower = lower_values[valid_mask]
        valid_upper = upper_values[valid_mask]
        
        # Find inverted intervals (lower > upper + tolerance)
        inverted_mask = valid_lower > (valid_upper + self.tolerance)
        
        # Get original indices of inverted intervals
        valid_indices = np.where(valid_mask)[0]
        inverted_valid_indices = np.where(inverted_mask)[0]
        inverted_indices = valid_indices[inverted_valid_indices]
        
        # Create detailed information
        details = []
        for idx in inverted_indices:
            details.append({
                'row': idx,
                'feature': feature,
                'lower_column': lower_col,
                'upper_column': upper_col,
                'lower_value': lower_values.iloc[idx],
                'upper_value': upper_values.iloc[idx],
                'difference': lower_values.iloc[idx] - upper_values.iloc[idx]
            })
        
        return {
            'count': len(inverted_indices),
            'rows': inverted_indices.tolist(),
            'details': details
        }
    
    def fix_inverted_intervals(self, interval_data: 'IntervalData', 
                             action: Union[str, InversionAction] = InversionAction.SWAP,
                             inplace: bool = False, 
                             return_format: str = 'intervaldata') -> Union['IntervalData', np.ndarray]:
        """
        Fix inverted intervals using the specified action
        
        Args:
            interval_data: IntervalData instance to fix
            action: Action to take (swap, remove, set_nan, set_equal, ignore)
            inplace: Whether to modify the original data or return a copy
            
        Returns:
            IntervalData instance with fixed intervals
        """
        if isinstance(action, str):
            try:
                action = InversionAction(action.lower())
            except ValueError:
                raise ValueError(f"Invalid action: {action}. Must be one of: {[e.value for e in InversionAction]}")
        
        # Work on copy unless inplace=True
        if inplace:
            result_data = interval_data
        else:
            result_data = interval_data.copy()
        
        # Check for inverted intervals
        check_results = self.check_intervals(interval_data)
        
        if not check_results['has_inverted']:
            print("No inverted intervals found.")
            return result_data
        
        if action == InversionAction.IGNORE:
            print(f"Ignoring {check_results['total_inverted']} inverted intervals.")
            return result_data
        
        # Apply the specified action
        if action == InversionAction.SWAP:
            result_data = self._swap_inverted_intervals(result_data, check_results)
        elif action == InversionAction.REMOVE:
            result_data = self._remove_inverted_rows(result_data, check_results)
        elif action == InversionAction.SET_NAN:
            result_data = self._set_inverted_to_nan(result_data, check_results)
        
        return result_data
    
    def _swap_inverted_intervals(self, interval_data: 'IntervalData', 
                               check_results: Dict) -> 'IntervalData':
        """Swap lower and upper bounds for inverted intervals"""
        data_copy = interval_data.data.copy()
        swapped_count = 0
        
        for detail in check_results['inverted_details']:
            row = detail['row']
            lower_col = detail['lower_column']
            upper_col = detail['upper_column']
            
            # Swap the values
            lower_val = data_copy.at[row, lower_col]
            upper_val = data_copy.at[row, upper_col]
            
            data_copy.at[row, lower_col] = upper_val
            data_copy.at[row, upper_col] = lower_val
            swapped_count += 1
        
        print(f"Swapped bounds for {swapped_count} inverted intervals.")
        
        # Return new IntervalData instance
        return IntervalData(data_copy)
    
    def _remove_inverted_rows(self, interval_data: 'IntervalData', 
                            check_results: Dict) -> 'IntervalData':
        """Remove rows containing inverted intervals"""
        data_copy = interval_data.data.copy()
        rows_to_remove = check_results['inverted_rows']
        
        if rows_to_remove:
            data_copy = data_copy.drop(index=rows_to_remove).reset_index(drop=True)
            print(f"Removed {len(rows_to_remove)} rows with inverted intervals.")
        
        from __main__ import IntervalData
        return IntervalData(data_copy)
    
    def _set_inverted_to_nan(self, interval_data: 'IntervalData', 
                           check_results: Dict) -> 'IntervalData':
        """Set inverted interval bounds to NaN"""
        data_copy = interval_data.data.copy()
        nan_count = 0
        
        for detail in check_results['inverted_details']:
            row = detail['row']
            lower_col = detail['lower_column']
            upper_col = detail['upper_column']
            
            data_copy.at[row, lower_col] = np.nan
            data_copy.at[row, upper_col] = np.nan
            nan_count += 1
        
        print(f"Set {nan_count} inverted intervals to NaN.")
        
        from __main__ import IntervalData
        return IntervalData(data_copy)
    
    def print_report(self, interval_data: 'IntervalData', 
                    detailed: bool = False) -> None:
        """
        Print a simple report about inverted intervals
        
        Args:
            interval_data: IntervalData instance to analyze
            detailed: Whether to show detailed information for each inverted interval
        """
        results = self.check_intervals(interval_data)
        
        print("=== Inverted Interval Analysis Report ===")
        print(f"Total samples: {len(interval_data)}")
        print(f"Total features: {interval_data.n_features}")
        print(f"Has inverted intervals: {results['has_inverted']}")
        print(f"Total inverted intervals: {results['total_inverted']}")
        
        if results['has_inverted']:
            print(f"Affected rows: {len(results['inverted_rows'])}")
            print(f"Percentage of affected samples: {len(results['inverted_rows'])/len(interval_data)*100:.2f}%")
            
            print("\nInverted intervals by feature:")
            for feature, count in results['inverted_by_feature'].items():
                percentage = count / len(interval_data) * 100
                print(f"  {feature}: {count} ({percentage:.2f}%)")
            
            if detailed and results['inverted_details']:
                print("\nDetailed inverted interval information:")
                for i, detail in enumerate(results['inverted_details'][:10], 1):  # Show first 10
                    print(f"  {i:2d}. Row {detail['row']:3d}, Feature '{detail['feature']}': "
                          f"{detail['lower_value']:.4f} > {detail['upper_value']:.4f} "
                          f"(diff: {detail['difference']:.4f})")
                
                if len(results['inverted_details']) > 10:
                    print(f"  ... and {len(results['inverted_details']) - 10} more")
        else:
            print("✓ No inverted intervals detected!")
    
    def has_missing_values(self, interval_data: 'IntervalData') -> bool:
        """
        Quick check if the data contains any missing values in interval columns
        
        Args:
            interval_data: IntervalData instance to check
            
        Returns:
            bool: True if any missing values found in interval columns
        """
        for lower_col, upper_col in interval_data.interval_pairs:
            if (interval_data.data[lower_col].isna().any() or 
                interval_data.data[upper_col].isna().any()):
                return True
        return False
        """Get results from the last check_intervals call"""
        return self._last_check_results
    
    @staticmethod
    def validate_and_fix(interval_data: 'IntervalData', 
                        action: Union[str, InversionAction] = InversionAction.SWAP,
                        tolerance: float = 1e-10,
                        show_report: bool = True) -> 'IntervalData':
        """
        Convenience method to check and fix inverted intervals in one call
        
        Args:
            interval_data: IntervalData instance to process
            action: Action to take for inverted intervals
            tolerance: Tolerance for numerical comparison
            show_report: Whether to print a report
            
        Returns:
            IntervalData instance with fixed intervals
        """
        checker = InvertedIntervalHandler(tolerance=tolerance)
        
        if show_report:
            checker.print_report(interval_data)
        
        return checker.fix_inverted_intervals(interval_data, action=action)
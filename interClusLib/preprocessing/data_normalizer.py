from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, Tuple
from interClusLib.IntervalData import IntervalData


class MinMaxNormalizer:
    """
    Min-Max normalization for IntervalData while preserving interval relationships.
    
    Parameters
    ----------
    feature_range : tuple, default=(0, 1)
        Target range for normalization.
    """
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.global_min_ = {}
        self.global_max_ = {}
        self.fitted_ = False
        self._last_check_results = None
    
    def fit(self, interval_data):
        """
        Fit the normalizer to IntervalData.
        
        Parameters
        ----------
        interval_data : IntervalData
            Interval data instance
            
        Returns
        -------
        self : MinMaxNormalizer
            Fitted normalizer
        """
        for feature in interval_data.features:
            lower_col, upper_col = self._get_feature_columns(interval_data, feature)
            
            if lower_col and upper_col:
                lower_values = interval_data.data[lower_col]
                upper_values = interval_data.data[upper_col]
                
                # Skip NaN values
                valid_lower = lower_values[pd.notna(lower_values)]
                valid_upper = upper_values[pd.notna(upper_values)]
                
                if len(valid_lower) > 0 and len(valid_upper) > 0:
                    self.global_min_[feature] = min(valid_lower.min(), valid_upper.min())
                    self.global_max_[feature] = max(valid_lower.max(), valid_upper.max())
                else:
                    # Handle case where all values are NaN
                    self.global_min_[feature] = 0.0
                    self.global_max_[feature] = 1.0
        
        self.fitted_ = True
        return self
    
    def transform(self, interval_data, inplace=False):
        """
        Transform IntervalData using fitted parameters.
        
        Parameters
        ----------
        interval_data : IntervalData
            Interval data to transform
        inplace : bool, default=False
            Whether to modify the original data
            
        Returns
        -------
        IntervalData
            Normalized interval data
        """
        if not self.fitted_:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        if inplace:
            result_data = interval_data
        else:
            result_data = interval_data.copy()
        
        data_copy = result_data.data.copy()
        min_new, max_new = self.feature_range
        
        for feature in interval_data.features:
            lower_col, upper_col = self._get_feature_columns(interval_data, feature)
            
            if lower_col and upper_col and feature in self.global_min_:
                global_min = self.global_min_[feature]
                global_max = self.global_max_[feature]
                range_val = global_max - global_min
                
                if range_val == 0:
                    # Handle constant values
                    data_copy[lower_col] = min_new
                    data_copy[upper_col] = min_new
                else:
                    # Transform lower bounds
                    valid_lower_mask = pd.notna(data_copy[lower_col])
                    data_copy.loc[valid_lower_mask, lower_col] = (
                        min_new + (data_copy.loc[valid_lower_mask, lower_col] - global_min) * 
                        (max_new - min_new) / range_val
                    )
                    
                    # Transform upper bounds
                    valid_upper_mask = pd.notna(data_copy[upper_col])
                    data_copy.loc[valid_upper_mask, upper_col] = (
                        min_new + (data_copy.loc[valid_upper_mask, upper_col] - global_min) * 
                        (max_new - min_new) / range_val
                    )
        
        if inplace:
            result_data.data = data_copy
            return result_data
        else:
            return IntervalData(data_copy)
    
    def fit_transform(self, interval_data, inplace=False):
        """
        Fit the normalizer and transform the data in one step.
        """
        return self.fit(interval_data).transform(interval_data, inplace=inplace)
    
    def inverse_transform(self, normalized_data):
        """
        Inverse transform normalized data back to original scale.
        """
        if not self.fitted_:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")
        
        data_copy = normalized_data.data.copy()
        min_new, max_new = self.feature_range
        
        for feature in normalized_data.features:
            lower_col, upper_col = self._get_feature_columns(normalized_data, feature)
            
            if lower_col and upper_col and feature in self.global_min_:
                global_min = self.global_min_[feature]
                global_max = self.global_max_[feature]
                range_val = global_max - global_min
                
                if range_val != 0:
                    # Inverse transform lower bounds
                    valid_lower_mask = pd.notna(data_copy[lower_col])
                    data_copy.loc[valid_lower_mask, lower_col] = (
                        global_min + (data_copy.loc[valid_lower_mask, lower_col] - min_new) * 
                        range_val / (max_new - min_new)
                    )
                    
                    # Inverse transform upper bounds
                    valid_upper_mask = pd.notna(data_copy[upper_col])
                    data_copy.loc[valid_upper_mask, upper_col] = (
                        global_min + (data_copy.loc[valid_upper_mask, upper_col] - min_new) * 
                        range_val / (max_new - min_new)
                    )
                else:
                    # Restore constant values
                    data_copy[lower_col] = global_min
                    data_copy[upper_col] = global_min
        
        return IntervalData(data_copy)
    
    def print_report(self, interval_data):
        """Print a report about the Min-Max normalization"""
        print("=== Min-Max Normalization Report ===")
        print(f"Target range: {self.feature_range}")
        print(f"Total samples: {interval_data.n_samples}")
        print(f"Total features: {interval_data.n_features}")
        
        if self.fitted_:
            print("✓ Normalizer is fitted and ready")
            print("Feature ranges:")
            for feature in self.global_min_:
                min_val = self.global_min_[feature]
                max_val = self.global_max_[feature]
                range_val = max_val - min_val
                print(f"  {feature}: [{min_val:.4f}, {max_val:.4f}] (range: {range_val:.4f})")
        else:
            print("⚠ Normalizer not fitted yet")
    
    def _get_feature_columns(self, interval_data, feature: str) -> Tuple[Optional[str], Optional[str]]:
        """Get lower and upper column names for a feature"""
        for lower, upper in interval_data.interval_pairs:
            if feature in lower and feature in upper:
                return lower, upper
        return None, None


class ZScoreNormalizer:
    """
    Z-score normalization (standardization) for IntervalData.
    Standardizes midpoints and maintains relative widths.
    """
    
    def __init__(self):
        self.mid_mean_ = {}
        self.mid_std_ = {}
        self.fitted_ = False
        self._last_check_results = None
    
    def fit(self, interval_data):
        """
        Fit the normalizer to IntervalData.
        """
        for feature in interval_data.features:
            lower_col, upper_col = self._get_feature_columns(interval_data, feature)
            
            if lower_col and upper_col:
                lower_values = interval_data.data[lower_col]
                upper_values = interval_data.data[upper_col]
                
                # Calculate midpoints for valid intervals only
                valid_mask = pd.notna(lower_values) & pd.notna(upper_values)
                
                if valid_mask.any():
                    valid_lower = lower_values[valid_mask]
                    valid_upper = upper_values[valid_mask]
                    midpoints = (valid_lower + valid_upper) / 2
                    
                    self.mid_mean_[feature] = midpoints.mean()
                    self.mid_std_[feature] = midpoints.std(ddof=1)
                    
                    if self.mid_std_[feature] == 0:
                        self.mid_std_[feature] = 1.0  # Avoid division by zero
                else:
                    # Handle case where all values are NaN
                    self.mid_mean_[feature] = 0.0
                    self.mid_std_[feature] = 1.0
        
        self.fitted_ = True
        return self
    
    def transform(self, interval_data, inplace=False):
        """
        Transform IntervalData using fitted parameters.
        """
        if not self.fitted_:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        if inplace:
            result_data = interval_data
        else:
            result_data = interval_data.copy()
        
        data_copy = result_data.data.copy()
        
        for feature in interval_data.features:
            lower_col, upper_col = self._get_feature_columns(interval_data, feature)
            
            if lower_col and upper_col and feature in self.mid_mean_:
                lower_values = data_copy[lower_col]
                upper_values = data_copy[upper_col]
                
                # Only transform valid intervals
                valid_mask = pd.notna(lower_values) & pd.notna(upper_values)
                
                if valid_mask.any():
                    valid_lower = lower_values[valid_mask]
                    valid_upper = upper_values[valid_mask]
                    
                    midpoints = (valid_lower + valid_upper) / 2
                    widths = valid_upper - valid_lower
                    
                    # Standardize midpoints
                    norm_midpoints = (midpoints - self.mid_mean_[feature]) / self.mid_std_[feature]
                    # Scale widths proportionally
                    norm_widths = widths / (self.mid_std_[feature] * 2)
                    
                    # Reconstruct intervals
                    data_copy.loc[valid_mask, lower_col] = norm_midpoints - norm_widths / 2
                    data_copy.loc[valid_mask, upper_col] = norm_midpoints + norm_widths / 2
        
        if inplace:
            result_data.data = data_copy
            return result_data
        else:
            return IntervalData(data_copy)
    
    def fit_transform(self, interval_data, inplace=False):
        """
        Fit the normalizer and transform the data in one step.
        """
        return self.fit(interval_data).transform(interval_data, inplace=inplace)
    
    def inverse_transform(self, normalized_data):
        """
        Inverse transform standardized data back to original scale.
        """
        if not self.fitted_:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")
        
        data_copy = normalized_data.data.copy()
        
        for feature in normalized_data.features:
            lower_col, upper_col = self._get_feature_columns(normalized_data, feature)
            
            if lower_col and upper_col and feature in self.mid_mean_:
                lower_values = data_copy[lower_col]
                upper_values = data_copy[upper_col]
                
                valid_mask = pd.notna(lower_values) & pd.notna(upper_values)
                
                if valid_mask.any():
                    valid_lower = lower_values[valid_mask]
                    valid_upper = upper_values[valid_mask]
                    
                    norm_midpoints = (valid_lower + valid_upper) / 2
                    norm_widths = valid_upper - valid_lower
                    
                    # Inverse transform
                    midpoints = norm_midpoints * self.mid_std_[feature] + self.mid_mean_[feature]
                    widths = norm_widths * (self.mid_std_[feature] * 2)
                    
                    # Reconstruct intervals
                    data_copy.loc[valid_mask, lower_col] = midpoints - widths / 2
                    data_copy.loc[valid_mask, upper_col] = midpoints + widths / 2
        
        return IntervalData(data_copy)
    
    def print_report(self, interval_data):
        """Print a report about the Z-Score normalization"""
        print("=== Z-Score Normalization Report ===")
        print(f"Total samples: {interval_data.n_samples}")
        print(f"Total features: {interval_data.n_features}")
        
        if self.fitted_:
            print("✓ Normalizer is fitted and ready")
            print("Standardization parameters:")
            for feature in self.mid_mean_:
                mean_val = self.mid_mean_[feature]
                std_val = self.mid_std_[feature]
                print(f"  {feature}: mean={mean_val:.4f}, std={std_val:.4f}")
        else:
            print("⚠ Normalizer not fitted yet")
    
    def _get_feature_columns(self, interval_data, feature: str) -> Tuple[Optional[str], Optional[str]]:
        """Get lower and upper column names for a feature"""
        for lower, upper in interval_data.interval_pairs:
            if feature in lower and feature in upper:
                return lower, upper
        return None, None


class MidpointWidthNormalizer:
    """
    Normalize the midpoint and width of intervals separately for IntervalData.
    """
    
    def __init__(self):
        self.mid_min_ = {}
        self.mid_max_ = {}
        self.width_min_ = {}
        self.width_max_ = {}
        self.fitted_ = False
        self._last_check_results = None
    
    def fit(self, interval_data):
        """
        Fit the normalizer to IntervalData.
        """
        for feature in interval_data.features:
            lower_col, upper_col = self._get_feature_columns(interval_data, feature)
            
            if lower_col and upper_col:
                lower_values = interval_data.data[lower_col]
                upper_values = interval_data.data[upper_col]
                
                # Calculate midpoints and widths for valid intervals only
                valid_mask = pd.notna(lower_values) & pd.notna(upper_values)
                
                if valid_mask.any():
                    valid_lower = lower_values[valid_mask]
                    valid_upper = upper_values[valid_mask]
                    midpoints = (valid_lower + valid_upper) / 2
                    widths = valid_upper - valid_lower
                    
                    self.mid_min_[feature] = midpoints.min()
                    self.mid_max_[feature] = midpoints.max()
                    self.width_min_[feature] = widths.min()
                    self.width_max_[feature] = widths.max()
                else:
                    # Handle case where all values are NaN
                    self.mid_min_[feature] = 0.0
                    self.mid_max_[feature] = 1.0
                    self.width_min_[feature] = 0.0
                    self.width_max_[feature] = 1.0
        
        self.fitted_ = True
        return self
    
    def transform(self, interval_data, inplace=False):
        """
        Transform IntervalData using fitted parameters.
        """
        if not self.fitted_:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        if inplace:
            result_data = interval_data
        else:
            result_data = interval_data.copy()
        
        data_copy = result_data.data.copy()
        
        for feature in interval_data.features:
            lower_col, upper_col = self._get_feature_columns(interval_data, feature)
            
            if lower_col and upper_col and feature in self.mid_min_:
                lower_values = data_copy[lower_col]
                upper_values = data_copy[upper_col]
                
                # Only transform valid intervals
                valid_mask = pd.notna(lower_values) & pd.notna(upper_values)
                
                if valid_mask.any():
                    valid_lower = lower_values[valid_mask]
                    valid_upper = upper_values[valid_mask]
                    
                    midpoints = (valid_lower + valid_upper) / 2
                    widths = valid_upper - valid_lower
                    
                    # Normalize midpoints
                    mid_range = self.mid_max_[feature] - self.mid_min_[feature]
                    if mid_range == 0:
                        norm_midpoints = np.zeros_like(midpoints)
                    else:
                        norm_midpoints = (midpoints - self.mid_min_[feature]) / mid_range
                    
                    # Normalize widths
                    width_range = self.width_max_[feature] - self.width_min_[feature]
                    if width_range == 0:
                        norm_widths = np.zeros_like(widths)
                    else:
                        norm_widths = (widths - self.width_min_[feature]) / width_range
                    
                    # Reconstruct intervals
                    data_copy.loc[valid_mask, lower_col] = norm_midpoints - norm_widths / 2
                    data_copy.loc[valid_mask, upper_col] = norm_midpoints + norm_widths / 2
        
        if inplace:
            result_data.data = data_copy
            return result_data
        else:
            return IntervalData(data_copy)
    
    def fit_transform(self, interval_data, inplace=False):
        """
        Fit the normalizer and transform the data in one step.
        """
        return self.fit(interval_data).transform(interval_data, inplace=inplace)
    
    def inverse_transform(self, normalized_data):
        """
        Inverse transform normalized data back to original scale.
        """
        if not self.fitted_:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")
        
        data_copy = normalized_data.data.copy()
        
        for feature in normalized_data.features:
            lower_col, upper_col = self._get_feature_columns(normalized_data, feature)
            
            if lower_col and upper_col and feature in self.mid_min_:
                lower_values = data_copy[lower_col]
                upper_values = data_copy[upper_col]
                
                valid_mask = pd.notna(lower_values) & pd.notna(upper_values)
                
                if valid_mask.any():
                    valid_lower = lower_values[valid_mask]
                    valid_upper = upper_values[valid_mask]
                    
                    norm_midpoints = (valid_lower + valid_upper) / 2
                    norm_widths = valid_upper - valid_lower
                    
                    # Denormalize midpoints
                    mid_range = self.mid_max_[feature] - self.mid_min_[feature]
                    if mid_range == 0:
                        midpoints = pd.Series(np.full_like(norm_midpoints, self.mid_min_[feature]), index=norm_midpoints.index)
                    else:
                        midpoints = norm_midpoints * mid_range + self.mid_min_[feature]
                    
                    # Denormalize widths
                    width_range = self.width_max_[feature] - self.width_min_[feature]
                    if width_range == 0:
                        widths = pd.Series(np.full_like(norm_widths, self.width_min_[feature]), index=norm_widths.index)
                    else:
                        widths = norm_widths * width_range + self.width_min_[feature]
                    
                    # Reconstruct intervals
                    data_copy.loc[valid_mask, lower_col] = midpoints - widths / 2
                    data_copy.loc[valid_mask, upper_col] = midpoints + widths / 2
        
        return IntervalData(data_copy)
    
    def print_report(self, interval_data):
        """Print a report about the Midpoint-Width normalization"""
        print("=== Midpoint-Width Normalization Report ===")
        print(f"Total samples: {interval_data.n_samples}")
        print(f"Total features: {interval_data.n_features}")
        
        if self.fitted_:
            print("✓ Normalizer is fitted and ready")
            print("Normalization parameters:")
            for feature in self.mid_min_:
                mid_range = self.mid_max_[feature] - self.mid_min_[feature]
                width_range = self.width_max_[feature] - self.width_min_[feature]
                print(f"  {feature}:")
                print(f"    Midpoint: [{self.mid_min_[feature]:.4f}, {self.mid_max_[feature]:.4f}] (range: {mid_range:.4f})")
                print(f"    Width: [{self.width_min_[feature]:.4f}, {self.width_max_[feature]:.4f}] (range: {width_range:.4f})")
        else:
            print("⚠ Normalizer not fitted yet")
    
    def _get_feature_columns(self, interval_data, feature: str) -> Tuple[Optional[str], Optional[str]]:
        """Get lower and upper column names for a feature"""
        for lower, upper in interval_data.interval_pairs:
            if feature in lower and feature in upper:
                return lower, upper
        return None, None


class RobustQuantileNormalizer:
    """
    Robust normalization using quantiles to handle outliers for IntervalData.
    
    Parameters
    ----------
    q_low : float, default=0.05
        Lower quantile threshold (5th percentile).
    q_high : float, default=0.95
        Upper quantile threshold (95th percentile).
    """
    
    def __init__(self, q_low=0.05, q_high=0.95):
        self.q_low = q_low
        self.q_high = q_high
        self.min_q_ = {}
        self.max_q_ = {}
        self.fitted_ = False
        self._last_check_results = None
    
    def fit(self, interval_data):
        """
        Fit the normalizer to IntervalData.
        """
        for feature in interval_data.features:
            lower_col, upper_col = self._get_feature_columns(interval_data, feature)
            
            if lower_col and upper_col:
                lower_values = interval_data.data[lower_col]
                upper_values = interval_data.data[upper_col]
                
                # Skip NaN values
                valid_lower = lower_values[pd.notna(lower_values)]
                valid_upper = upper_values[pd.notna(upper_values)]
                
                if len(valid_lower) > 0 and len(valid_upper) > 0:
                    self.min_q_[feature] = np.quantile(valid_lower, self.q_low)
                    self.max_q_[feature] = np.quantile(valid_upper, self.q_high)
                else:
                    # Handle case where all values are NaN
                    self.min_q_[feature] = 0.0
                    self.max_q_[feature] = 1.0
        
        self.fitted_ = True
        return self
    
    def transform(self, interval_data, inplace=False):
        """
        Transform IntervalData using fitted parameters.
        """
        if not self.fitted_:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        if inplace:
            result_data = interval_data
        else:
            result_data = interval_data.copy()
        
        data_copy = result_data.data.copy()
        
        for feature in interval_data.features:
            lower_col, upper_col = self._get_feature_columns(interval_data, feature)
            
            if lower_col and upper_col and feature in self.min_q_:
                min_q = self.min_q_[feature]
                max_q = self.max_q_[feature]
                range_q = max_q - min_q
                
                if range_q == 0:
                    # Handle constant values
                    valid_lower_mask = pd.notna(data_copy[lower_col])
                    valid_upper_mask = pd.notna(data_copy[upper_col])
                    data_copy.loc[valid_lower_mask, lower_col] = 0.0
                    data_copy.loc[valid_upper_mask, upper_col] = 0.0
                else:
                    # Transform and clip lower bounds
                    valid_lower_mask = pd.notna(data_copy[lower_col])
                    data_copy.loc[valid_lower_mask, lower_col] = np.clip(
                        (data_copy.loc[valid_lower_mask, lower_col] - min_q) / range_q, 0, 1
                    )
                    
                    # Transform and clip upper bounds
                    valid_upper_mask = pd.notna(data_copy[upper_col])
                    data_copy.loc[valid_upper_mask, upper_col] = np.clip(
                        (data_copy.loc[valid_upper_mask, upper_col] - min_q) / range_q, 0, 1
                    )
        
        if inplace:
            result_data.data = data_copy
            return result_data
        else:
            return IntervalData(data_copy)
    
    def fit_transform(self, interval_data, inplace=False):
        """
        Fit the normalizer and transform the data in one step.
        """
        return self.fit(interval_data).transform(interval_data, inplace=inplace)
    
    def inverse_transform(self, normalized_data):
        """
        Inverse transform normalized data back to original scale.
        """
        if not self.fitted_:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")
        
        data_copy = normalized_data.data.copy()
        
        for feature in normalized_data.features:
            lower_col, upper_col = self._get_feature_columns(normalized_data, feature)
            
            if lower_col and upper_col and feature in self.min_q_:
                min_q = self.min_q_[feature]
                max_q = self.max_q_[feature]
                range_q = max_q - min_q
                
                if range_q != 0:
                    # Inverse transform lower bounds
                    valid_lower_mask = pd.notna(data_copy[lower_col])
                    data_copy.loc[valid_lower_mask, lower_col] = (
                        data_copy.loc[valid_lower_mask, lower_col] * range_q + min_q
                    )
                    
                    # Inverse transform upper bounds
                    valid_upper_mask = pd.notna(data_copy[upper_col])
                    data_copy.loc[valid_upper_mask, upper_col] = (
                        data_copy.loc[valid_upper_mask, upper_col] * range_q + min_q
                    )
                else:
                    # Restore constant values
                    data_copy[lower_col] = min_q
                    data_copy[upper_col] = min_q
        
        return IntervalData(data_copy)
    
    def print_report(self, interval_data):
        """Print a report about the Robust Quantile normalization"""
        print("=== Robust Quantile Normalization Report ===")
        print(f"Quantile range: [{self.q_low:.1%}, {self.q_high:.1%}]")
        print(f"Total samples: {interval_data.n_samples}")
        print(f"Total features: {interval_data.n_features}")
        
        if self.fitted_:
            print("✓ Normalizer is fitted and ready")
            print("Quantile ranges:")
            for feature in self.min_q_:
                min_q = self.min_q_[feature]
                max_q = self.max_q_[feature]
                range_q = max_q - min_q
                print(f"  {feature}: [{min_q:.4f}, {max_q:.4f}] (range: {range_q:.4f})")
        else:
            print("⚠ Normalizer not fitted yet")
    
    def _get_feature_columns(self, interval_data, feature: str) -> Tuple[Optional[str], Optional[str]]:
        """Get lower and upper column names for a feature"""
        for lower, upper in interval_data.interval_pairs:
            if feature in lower and feature in upper:
                return lower, upper
        return None, None
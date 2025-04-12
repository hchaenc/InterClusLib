import pandas as pd
import numpy as np
import os
import re

class IntervalData:
    """
    Handles interval data, supports CSV and XLSX loading while preserving original column names
    Added support for handling missing values (NaN)
    """
    def __init__(self, data, handle_missing=True):
        """
        :param data: Pandas DataFrame
        :param handle_missing: Whether to handle missing values or raise an error
        """
        self.handle_missing = handle_missing
        self.data = self._validate_data(data)
        # Add columns and features attributes
        self.columns = self._get_all_columns()
        self.features = self._extract_feature_names()
        # Keep the has_missing_values attribute
        self.has_missing_values = self._check_missing_values()

    def _validate_data(self, data):
        """ Ensures that the input data is a Pandas DataFrame """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a Pandas DataFrame!")
        
        # Report on missing values but don't fail
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            print(f"Warning: DataFrame contains {missing_values} missing values.")
            print(data.isnull().sum())
            
            if not self.handle_missing:
                raise ValueError("DataFrame contains missing values. Set handle_missing=True to proceed anyway.")
        
        return data

    def _check_missing_values(self):
        """
        Check if the data contains missing values and return True/False
        Returns a Python native bool, not a numpy.bool_
        """
        # Directly check if there are missing values in the entire DataFrame
        missing_count = self.data.isnull().sum().sum()
        
        # Use bool() to ensure returning a Python native boolean, not numpy.bool_
        return bool(missing_count > 0)

    def _detect_intervals(self):
        """ Automatically detects paired interval columns while preserving original column names """
        columns = self.data.columns.tolist()
        interval_pairs = []

        # Regex pattern to match columns with '_lower' and '_upper' suffixes
        pattern = re.compile(r"(.*)_(lower|upper)", re.IGNORECASE)
        col_groups = {}

        for col in columns:
            match = pattern.match(col)
            if match:
                base_name = match.group(1)
                col_type = match.group(2)
                if base_name not in col_groups:
                    col_groups[base_name] = {}
                col_groups[base_name][col_type] = col

        # Ensure both 'lower' and 'upper' exist in each interval pair
        for base_name, cols in col_groups.items():
            if "lower" in cols and "upper" in cols:
                interval_pairs.append((cols["lower"], cols["upper"]))

        if len(interval_pairs) == 0:
            raise ValueError("No valid interval columns detected. Ensure columns are formatted as '_lower' and '_upper'.")

        return interval_pairs
    
    def _get_all_columns(self):
        """
        Returns a list of all column names in the data
        Returns a list of column names like [feature_1_lower, feature_1_upper, feature_2_lower]
        """
        return self.data.columns.tolist()
    
    def _extract_feature_names(self):
        """
        Extracts the base feature names from all columns
        Returns a list of feature names like [feature_1, feature_2]
        """
        feature_names = []
        pattern = re.compile(r"(.*)_(lower|upper)", re.IGNORECASE)
        
        for col in self.columns:
            match = pattern.match(col)
            if match:
                feature_name = match.group(1)
                if feature_name not in feature_names:
                    feature_names.append(feature_name)
        
        return feature_names

    @classmethod
    def from_csv(cls, file_path, handle_missing=True):
        """ Loads interval data from a CSV file """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist!")

        data = pd.read_csv(file_path)
        return cls(data, handle_missing=handle_missing)

    @classmethod
    def from_excel(cls, file_path, sheet_name=0, handle_missing=True):
        """ Loads interval data from an Excel file """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist!")

        data = pd.read_excel(file_path, sheet_name=sheet_name)
        return cls(data, handle_missing=handle_missing)

    @classmethod
    def random_data(cls, n, m, lower=0, upper=100):
        """
        Generate a random interval value dataset, with each interval value within the specified range.

        Parameters:
            n (int): Number of samples (rows).
            m (int): Number of features (columns).
            lower (int): Minimum lower limit of the interval value (default 0).
            upper (int): Maximum upper limit of the interval value (default 100).

        Returns:
            pd.DataFrame: Two-dimensional dataset containing interval values, with each column consisting of the interval start and end.
        """
        # create random interval data, with each interval value within the specified range
        start = np.random.uniform(lower, upper, (n, m))
        end = np.random.uniform(lower, upper, (n, m))

        # make sure the start value is less than the end
        interval_start = np.minimum(start, end)
        interval_end = np.maximum(start, end)

        data = {}
        # create a dataset with each column consisting of the interval start and end
        for i in range(m):
            data[f"Feature_{i+1}_lower"] = interval_start[:, i]
            data[f"Feature_{i+1}_upper"] = interval_end[:, i]

        return cls(pd.DataFrame(data))
        
    @classmethod
    def make_interval_blobs(cls,
                     n_samples=100,
                     n_clusters=3,
                     n_dims=2,
                     cluster_centers=None,
                     cluster_std_center=1.0,
                     cluster_std_width=0.5,
                     center_range=(-10, 10),    
                     width_range=(0.1, 2.0),    
                     min_width=0.05,            
                     shuffle=True,
                     random_state=None):
        """
        Generate synthetic interval data with cluster structure, similarly to random_data, but 
        arranged so that each sample belongs to one of several 'interval clusters'.

        :param n_samples: int or list of int (if list => number of samples per cluster)
        :param n_clusters: number of clusters
        :param n_dims: number of features => shape (n_dims,2)
        :param cluster_centers: None or array (n_clusters, n_dims,2)
        :param cluster_std_center: float or array-like controlling intervals' midpoint variation
        :param cluster_std_width: float or array-like controlling intervals' width variation
        :param center_range: tuple (min, max) for random center generation range
        :param width_range: tuple (min, max) for random width generation range
        :param min_width: minimum interval width (prevents negative or very small widths)
        :param shuffle: bool, shuffle the samples
        :param random_state: for reproducibility
        :return: an IntervalData object with columns [feature_1_lower, feature_1_upper, ...]
        """
        
        rng = np.random.default_rng(random_state)

        # 1) distribute n_samples among clusters
        if isinstance(n_samples, int):
            samples_per_cluster = [n_samples // n_clusters]*n_clusters
            remainder = n_samples % n_clusters
            for i in range(remainder):
                samples_per_cluster[i]+=1
        else:
            samples_per_cluster = n_samples
            n_clusters = len(samples_per_cluster)

        total_samples = sum(samples_per_cluster)

        # 2) create or use given cluster_centers => shape(n_clusters, n_dims, 2)
        if cluster_centers is None:
            # random centers with user-defined ranges
            cluster_centers = []
            for k in range(n_clusters):
                midpoint = rng.uniform(center_range[0], center_range[1], size=n_dims)
                half_width = rng.uniform(width_range[0], width_range[1], size=n_dims)
                lower = midpoint - half_width
                upper = midpoint + half_width
                cluster_centers.append(np.stack([lower, upper], axis=-1))
            cluster_centers = np.array(cluster_centers)
        else:
            cluster_centers = np.array(cluster_centers)
            # optionally check shape?

        cluster_std_center = np.array(cluster_std_center, ndmin=1)
        if cluster_std_center.size==1:
            cluster_std_center = np.full(n_dims, cluster_std_center[0])

        cluster_std_width = np.array(cluster_std_width, ndmin=1)
        if cluster_std_width.size==1:
            cluster_std_width = np.full(n_dims, cluster_std_width[0])

        intervals = np.zeros((total_samples, n_dims, 2), dtype=float)

        # 3) generate data
        current_idx=0
        for c_idx in range(n_clusters):
            n_k = samples_per_cluster[c_idx]
            c_center = cluster_centers[c_idx]  # shape (n_dims,2)

            # cluster midpoint / halfwidth
            midpoint = 0.5*(c_center[...,0] + c_center[...,1])
            halfw   = 0.5*(c_center[...,1] - c_center[...,0])

            # offset
            offset_mid = rng.normal(0, cluster_std_center, size=(n_k,n_dims))
            offset_hw  = rng.normal(0, cluster_std_width,  size=(n_k,n_dims))

            final_mid  = midpoint + offset_mid
            final_hw   = halfw + offset_hw
            final_hw[final_hw < min_width] = min_width  # Use user-defined minimum width

            a = final_mid - final_hw
            b = final_mid + final_hw

            intervals[current_idx:current_idx+n_k] = np.stack([a,b], axis=-1)
            current_idx+=n_k

        if shuffle:
            perm = rng.permutation(total_samples)
            intervals = intervals[perm]

        # 4) build the DataFrame
        df_data = {}
        for i in range(n_dims):
            df_data[f"Feature_{i+1}_lower"] = intervals[:, i, 0]
            df_data[f"Feature_{i+1}_upper"] = intervals[:, i, 1]

        df = pd.DataFrame(df_data)
        return cls(df)

    def get_intervals(self):
        """ 
        Returns interval data as a NumPy array in the shape [n_samples, n_intervals, 2]
        Modified to handle NaN values by either using a mask or safe conversion
        """
        # First need to get all interval pairs
        interval_pairs = []
        features = self.features
        
        for feature in features:
            lower_col = f"{feature}_lower"
            upper_col = f"{feature}_upper"
            if lower_col in self.columns and upper_col in self.columns:
                interval_pairs.append((lower_col, upper_col))
        
        if self.has_missing_values:
            # Method 1: Create masked array to preserve NaN information
            interval_data = []
            for lower, upper in interval_pairs:
                pair_data = self.data[[lower, upper]].to_numpy()
                interval_data.append(pair_data)
            
            # Convert to numpy array with NaN values preserved
            interval_data = np.array(interval_data).transpose((1, 0, 2))
            
            # Log warning about NaN values
            print(f"Warning: get_intervals() contains {np.isnan(interval_data).sum()} NaN values.")
            return interval_data
        else:
            # Original method for data without NaN
            interval_data = np.array([
                self.data[[lower, upper]].to_numpy() for lower, upper in interval_pairs
            ]).transpose((1, 0, 2))
            return interval_data

    def to_dataframe(self):
        """ Returns the original Pandas DataFrame """
        return self.data
    
    def validate_intervals(self, fix_invalid=False):
        """
        Validates that lower bounds are less than or equal to upper bounds.
        Optionally fixes invalid intervals by swapping values.
        
        :param fix_invalid: If True, swap values where lower > upper
        :return: DataFrame with validated/fixed intervals
        """
        fixed_data = self.data.copy()
        invalid_count = 0
        
        # Get all interval pairs
        interval_pairs = []
        features = self.features
        
        for feature in features:
            lower_col = f"{feature}_lower"
            upper_col = f"{feature}_upper"
            if lower_col in self.columns and upper_col in self.columns:
                interval_pairs.append((lower_col, upper_col))
        
        for lower, upper in interval_pairs:
            # Find rows where lower > upper (accounting for NaN values)
            mask = (fixed_data[lower] > fixed_data[upper]) & (~fixed_data[lower].isna()) & (~fixed_data[upper].isna())
            invalid_count += mask.sum()
            
            if fix_invalid and mask.sum() > 0:
                # Swap values for invalid intervals
                tmp = fixed_data.loc[mask, lower].copy()
                fixed_data.loc[mask, lower] = fixed_data.loc[mask, upper]
                fixed_data.loc[mask, upper] = tmp
                print(f"Fixed {mask.sum()} invalid intervals in {lower}/{upper}")
        
        if invalid_count > 0:
            if fix_invalid:
                return IntervalData(fixed_data)
            else:
                print(f"Warning: Found {invalid_count} intervals where lower > upper")
        
        return self if not fix_invalid else IntervalData(fixed_data)

    def summary(self):
        """ Prints basic statistical information and detected interval columns """
        print("Data Summary:")
        print(self.data.describe())
        
        # Get all interval pairs and print them
        interval_pairs = []
        features = self.features
        
        for feature in features:
            lower_col = f"{feature}_lower"
            upper_col = f"{feature}_upper"
            if lower_col in self.columns and upper_col in self.columns:
                interval_pairs.append((lower_col, upper_col))
        
        print("\nInterval column pairs:")
        for lower, upper in interval_pairs:
            print(f"- {lower} / {upper}")
        
        print("\nFeature names:")
        for feature in self.features:
            print(f"- {feature}")
            
        print("\nAll columns:")
        print(self.columns)
        
        print("\nHas missing values:")
        print(self.has_missing_values)
        
        if self.has_missing_values:
            print("\nMissing values per column:")
            print(self.data.isnull().sum())
    
    def save_to_csv(self, file_path):
        """ Saves the data to a CSV file """
        self.data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

    def save_to_excel(self, file_path):
        """ Saves the data to an Excel file """
        self.data.to_excel(file_path, index=False)
        print(f"Data saved to {file_path}")
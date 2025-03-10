import pandas as pd
import numpy as np
import os
import re

class IntervalData:
    """
    Handles interval data, supports CSV and XLSX loading while preserving original column names
    """
    def __init__(self, data):
        """
        :param data: Pandas DataFrame
        """
        self.data = self._validate_data(data)
        self.interval_columns = self._detect_intervals()

    def _validate_data(self, data):
        """ Ensures that the input data is a Pandas DataFrame """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a Pandas DataFrame!")
        return data

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

    @classmethod
    def from_csv(cls, file_path):
        """ Loads interval data from a CSV file """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist!")

        data = pd.read_csv(file_path)
        return cls(data)

    @classmethod
    def from_excel(cls, file_path, sheet_name=0):
        """ Loads interval data from an Excel file """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist!")

        data = pd.read_excel(file_path, sheet_name=sheet_name)
        return cls(data)

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
            data[f"feature_{i+1}_lower"] = interval_start[:, i]
            data[f"feature_{i+1}_upper"] = interval_end[:, i]

        return cls(pd.DataFrame(data))
        
    @classmethod
    def make_interval_blobs(cls,
                     n_samples=1000,
                     n_clusters=3,
                     n_dims=2,
                     cluster_centers=None,
                     cluster_std_center=1.0,
                     cluster_std_width=0.5,
                     center_range=(-10, 10),    # 新参数：控制中心点的随机范围
                     width_range=(0.1, 2.0),    # 新参数：控制宽度的随机范围
                     min_width=0.05,            # 新参数：最小宽度限制
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
            final_hw[final_hw < min_width] = min_width  # 使用用户定义的最小宽度

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
            df_data[f"feature_{i+1}_lower"] = intervals[:, i, 0]
            df_data[f"feature_{i+1}_upper"] = intervals[:, i, 1]

        df = pd.DataFrame(df_data)
        return cls(df)

    def get_intervals(self):
        """ Returns interval data as a NumPy array in the shape [n_samples, n_intervals, 2] """
        interval_data = np.array([
            self.data[[lower, upper]].to_numpy() for lower, upper in self.interval_columns
        ]).transpose((1, 0, 2))  # Reshape to (n_samples, n_intervals, 2)
        return interval_data

    def to_dataframe(self):
        """ Returns the original Pandas DataFrame """
        return self.data

    def summary(self):
        """ Prints basic statistical information and detected interval columns """
        print("Data Summary:")
        print(self.data.describe())
        print("\nDetected interval columns:")
        for lower, upper in self.interval_columns:
            print(f"- {lower} / {upper}")
    
    def save_to_csv(self, file_path):
        """ Saves the data to a CSV file """
        self.data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

    def save_to_excel(self, file_path):
        """ Saves the data to an Excel file """
        self.data.to_excel(file_path, index=False)
        print(f"Data saved to {file_path}")


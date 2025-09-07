import pandas as pd
import numpy as np
import os
import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union


class IntervalDataLoader(ABC):
    """Abstract base class for interval data loaders"""
    
    @abstractmethod
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """Load data from source"""
        pass


class CSVLoader(IntervalDataLoader):
    """CSV file loader"""
    
    def load(self, file_path: str, **kwargs) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist!")
        return pd.read_csv(file_path, **kwargs)


class ExcelLoader(IntervalDataLoader):
    """Excel file loader"""
    
    def load(self, file_path: str, sheet_name=0, **kwargs) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist!")
        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)


class IntervalColumnDetector:
    """Handles detection and validation of interval columns"""
    
    @staticmethod
    def detect_interval_pairs(columns: List[str]) -> List[Tuple[str, str]]:
        """Detect paired interval columns with _lower and _upper suffixes"""
        interval_pairs = []
        pattern = re.compile(r"(.*)_(lower|upper)", re.IGNORECASE)
        col_groups = {}

        for col in columns:
            match = pattern.match(col)
            if match:
                base_name = match.group(1)
                col_type = match.group(2).lower()
                if base_name not in col_groups:
                    col_groups[base_name] = {}
                col_groups[base_name][col_type] = col

        # Ensure both 'lower' and 'upper' exist in each interval pair
        for base_name, cols in col_groups.items():
            if "lower" in cols and "upper" in cols:
                interval_pairs.append((cols["lower"], cols["upper"]))

        if not interval_pairs:
            raise ValueError("No valid interval columns detected. Ensure columns are formatted as '_lower' and '_upper'.")

        return interval_pairs
    
    @staticmethod
    def extract_feature_names(columns: List[str]) -> List[str]:
        """Extract base feature names from interval columns"""
        feature_names = []
        pattern = re.compile(r"(.*)_(lower|upper)", re.IGNORECASE)
        
        for col in columns:
            match = pattern.match(col)
            if match:
                feature_name = match.group(1)
                if feature_name not in feature_names:
                    feature_names.append(feature_name)
        
        return feature_names

class IntervalDataGenerator:
    """Generates synthetic interval data"""
    
    @staticmethod
    def make_random_data(n_samples: int, n_features: int, 
                   lower: float = 0, upper: float = 100) -> pd.DataFrame:
        """
        Generate random interval data
        
        Args:
            n_samples: Number of samples (rows)
            n_features: Number of features (columns)
            lower: Minimum lower limit
            upper: Maximum upper limit
            
        Returns:
            DataFrame with interval columns
        """
        start = np.random.uniform(lower, upper, (n_samples, n_features))
        end = np.random.uniform(lower, upper, (n_samples, n_features))

        interval_start = np.minimum(start, end)
        interval_end = np.maximum(start, end)

        data = {}
        for i in range(n_features):
            data[f"Feature_{i+1}_lower"] = interval_start[:, i]
            data[f"Feature_{i+1}_upper"] = interval_end[:, i]

        return pd.DataFrame(data)
    
    @staticmethod
    def make_interval_blobs(n_samples: Union[int, List[int]] = 100,
                           n_clusters: int = 3,
                           n_dims: int = 2,
                           cluster_centers: Optional[np.ndarray] = None,
                           cluster_std_center: float = 1.0,
                           cluster_std_width: float = 0.5,
                           center_range: Tuple[float, float] = (-10, 10),    
                           width_range: Tuple[float, float] = (0.1, 2.0),    
                           min_width: float = 0.05,            
                           shuffle: bool = True,
                           random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic interval data with cluster structure
        """
        rng = np.random.default_rng(random_state)

        # Distribute samples among clusters
        if isinstance(n_samples, int):
            samples_per_cluster = [n_samples // n_clusters] * n_clusters
            remainder = n_samples % n_clusters
            for i in range(remainder):
                samples_per_cluster[i] += 1
        else:
            samples_per_cluster = n_samples
            n_clusters = len(samples_per_cluster)

        total_samples = sum(samples_per_cluster)

        # Create or use given cluster centers
        if cluster_centers is None:
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

        cluster_std_center = np.array(cluster_std_center, ndmin=1)
        if cluster_std_center.size == 1:
            cluster_std_center = np.full(n_dims, cluster_std_center[0])

        cluster_std_width = np.array(cluster_std_width, ndmin=1)
        if cluster_std_width.size == 1:
            cluster_std_width = np.full(n_dims, cluster_std_width[0])

        intervals = np.zeros((total_samples, n_dims, 2), dtype=float)

        # Generate data for each cluster
        current_idx = 0
        for c_idx in range(n_clusters):
            n_k = samples_per_cluster[c_idx]
            c_center = cluster_centers[c_idx]

            midpoint = 0.5 * (c_center[..., 0] + c_center[..., 1])
            halfw = 0.5 * (c_center[..., 1] - c_center[..., 0])

            offset_mid = rng.normal(0, cluster_std_center, size=(n_k, n_dims))
            offset_hw = rng.normal(0, cluster_std_width, size=(n_k, n_dims))

            final_mid = midpoint + offset_mid
            final_hw = halfw + offset_hw
            final_hw[final_hw < min_width] = min_width

            a = final_mid - final_hw
            b = final_mid + final_hw

            intervals[current_idx:current_idx+n_k] = np.stack([a, b], axis=-1)
            current_idx += n_k

        if shuffle:
            perm = rng.permutation(total_samples)
            intervals = intervals[perm]

        # Build DataFrame
        df_data = {}
        for i in range(n_dims):
            df_data[f"Feature_{i+1}_lower"] = intervals[:, i, 0]
            df_data[f"Feature_{i+1}_upper"] = intervals[:, i, 1]

        return pd.DataFrame(df_data)


class IntervalData:
    """
    Main class for handling interval data
    
    This class follows the Single Responsibility Principle by delegating
    specific tasks to specialized classes while maintaining a clean interface.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize IntervalData with a pandas DataFrame
        
        Args:
            data: Pandas DataFrame containing interval data
        """
        self._data = self._validate_data(data)
        self._detector = IntervalColumnDetector()
        
        # Cache computed properties
        self._interval_pairs = None
        self._features = None
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate input data"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a Pandas DataFrame!")
        return data
    
    @property
    def data(self) -> pd.DataFrame:
        """Get the underlying DataFrame"""
        return self._data
    
    @property 
    def columns(self) -> List[str]:
        """Get all column names"""
        return self._data.columns.tolist()
    
    @property
    def features(self) -> List[str]:
        """Get base feature names (cached)"""
        if self._features is None:
            self._features = self._detector.extract_feature_names(self.columns)
        return self._features
    
    @property
    def interval_pairs(self) -> List[Tuple[str, str]]:
        """Get interval column pairs (cached)"""
        if self._interval_pairs is None:
            self._interval_pairs = self._detector.detect_interval_pairs(self.columns)
        return self._interval_pairs
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get data shape"""
        return self._data.shape
    
    @property
    def n_samples(self) -> int:
        """Get number of samples"""
        return self._data.shape[0]
    
    @property
    def n_features(self) -> int:
        """Get number of interval features"""
        return len(self.features)
    
    # Factory methods for creating IntervalData instances
    @classmethod
    def from_csv(cls, file_path: str, **kwargs) -> 'IntervalData':
        """Create IntervalData from CSV file"""
        loader = CSVLoader()
        data = loader.load(file_path, **kwargs)
        return cls(data)

    @classmethod
    def from_excel(cls, file_path: str, sheet_name: Union[str, int] = 0, **kwargs) -> 'IntervalData':
        """Create IntervalData from Excel file"""
        loader = ExcelLoader()
        data = loader.load(file_path, sheet_name=sheet_name, **kwargs)
        return cls(data)

    @classmethod
    def make_random_data(cls, n_samples: int, n_features: int, 
                   lower: float = 0, upper: float = 100) -> 'IntervalData':
        """Create IntervalData with random intervals"""
        generator = IntervalDataGenerator()
        data = generator.make_random_data(n_samples, n_features, lower, upper)
        return cls(data)

    @classmethod
    def make_interval_blobs(cls, **kwargs) -> 'IntervalData':
        """Create IntervalData with clustered intervals"""
        generator = IntervalDataGenerator()
        data = generator.make_interval_blobs(**kwargs)
        return cls(data)
    
    # Core functionality methods
    def get_intervals(self) -> np.ndarray:
        """
        Get interval data as numpy array with shape [n_samples, n_features, 2]
        
        Returns:
            3D numpy array where the last dimension contains [lower, upper] bounds
        """
        interval_data = []
        for lower, upper in self.interval_pairs:
            pair_data = self._data[[lower, upper]].to_numpy()
            interval_data.append(pair_data)
        
        return np.array(interval_data).transpose((1, 0, 2))
    
    
    def validate_intervals(self, fix_invalid: bool = False) -> 'IntervalData':
        """
        Validate interval consistency - DEPRECATED
        
        This method is deprecated. Use preprocessing validation classes instead.
        
        Args:
            fix_invalid: Whether to fix invalid intervals by swapping bounds
            
        Returns:
            Self (unchanged)
        """
        import warnings
        warnings.warn(
            "IntervalData.validate_intervals() is deprecated. "
            "Use preprocessing validation classes instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return copy of the underlying DataFrame"""
        return self._data.copy()
    
    def copy(self) -> 'IntervalData':
        """Create a deep copy of this IntervalData instance"""
        return IntervalData(self._data.copy())
    
    # I/O methods
    def save_to_csv(self, file_path: str, **kwargs) -> None:
        """Save data to CSV file"""
        self._data.to_csv(file_path, index=False, **kwargs)
        print(f"Data saved to {file_path}")

    def save_to_excel(self, file_path: str, **kwargs) -> None:
        """Save data to Excel file"""
        self._data.to_excel(file_path, index=False, **kwargs)
        print(f"Data saved to {file_path}")
    
    # Information methods
    def summary(self) -> None:
        """Print comprehensive data summary"""
        print("=== IntervalData Summary ===")
        print(f"Shape: {self.shape}")
        print(f"Number of samples: {self.n_samples}")
        print(f"Number of interval features: {self.n_features}")
        
        print("\n=== Statistical Summary ===")
        print(self._data.describe())
        
        print("\n=== Interval Features ===")
        for i, feature in enumerate(self.features, 1):
            lower, upper = self.interval_pairs[i-1]
            print(f"{i:2d}. {feature:<20} -> {lower} / {upper}")
        
        print(f"\n=== Column Names ({len(self.columns)}) ===")
        for i, col in enumerate(self.columns, 1):
            print(f"{i:2d}. {col}")
    
    def info(self) -> None:
        """Print basic data information"""
        print("=== IntervalData Info ===")
        print(f"Shape: {self.shape}")
        print(f"Features: {self.n_features}")
        print(f"Samples: {self.n_samples}")
        print(f"Memory usage: {self._data.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Magic methods for better usability
    def __len__(self) -> int:
        """Return number of samples"""
        return self.n_samples
    
    def __str__(self) -> str:
        """String representation"""
        return f"IntervalData(samples={self.n_samples}, features={self.n_features})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"IntervalData(samples={self.n_samples}, features={self.n_features}, "
                f"columns={len(self.columns)})")
    
    def __getitem__(self, key) -> pd.DataFrame:
        """Allow indexing like a DataFrame"""
        return self._data[key]
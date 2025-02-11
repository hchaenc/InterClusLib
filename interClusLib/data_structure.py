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


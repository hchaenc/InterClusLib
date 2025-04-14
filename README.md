# INTERCLUSLIB

InterClusLib is a Python library focused on interval data clustering analysis and visualization. It provides various clustering algorithms, preprocessing tools, evaluation methods, and visualization capabilities.

## Features

- Multiple interval data clustering algorithms (K-Means, Fuzzy C-Means, Hierarchical Clustering, etc.)
- Complete data preprocessing toolkit (normalization, outlier detection, imputation)
- Various clustering evaluation metrics and methods (Elbow Method, Gap Statistic, L Method, etc.)
- Rich visualization tools (Heat Maps, Dendrograms, 2D/3D Interval Plots, Radar Charts, Parallel Coordinates)
- Flexible distance metrics and similarity calculations
- Detailed examples and documentation

## Installation

Install the latest stable version using pip:

```bash
pip install intercluslib
```

Install the development version from source:

```bash
git clone https://github.com/username/intercluslib.git
cd intercluslib
pip install -e .
```

### Dependencies

- Python (>=3.6)
- NumPy (>=1.17.0)
- pandas (>=1.0.0)
- scikit-learn (>=0.23.0)
- matplotlib (>=3.3.0)
- scipy (>=1.5.0)

## Quick Start

Here's a simple example demonstrating how to use INTERCLUSLIB for interval data clustering analysis:

```python
import intercluslib as icl
import pandas as pd
import matplotlib.pyplot as plt

# Load example data
data = pd.read_csv("examples/ChinaTemp.csv")
print(f"Data shape: {data.shape}")

# Data preprocessing
normalizer = icl.preprocessing.normalizer.MinMaxNormalizer()
normalized_data = normalizer.fit_transform(data)

# Perform clustering using Interval K-Means
kmeans = icl.clustering.IntervalKMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(normalized_data)

# Evaluate optimal number of clusters
evaluator = icl.evaluation.ElbowMethodEvaluator()
k_range = range(2, 10)
inertias = evaluator.compute_inertia(normalized_data, k_range)

# Visualize evaluation results
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

# Visualize clustering results
visualizer = icl.visualization.HeatMap()
visualizer.plot(normalized_data, clusters, title='Interval K-Means Clustering Results')
```

## Detailed Features

### Clustering Algorithms (`clustering` module)

InterClusLib provides several clustering algorithms specifically designed for interval data:

#### Interval K-Means

```python
from intercluslib.clustering import IntervalKMeans

# Create K-Means model and fit data
kmeans = IntervalKMeans(n_clusters=4, max_iter=300, random_state=42)
kmeans.fit(data)

print(kmeans.labels_)

# Get cluster centers
centers = kmeans.cluster_centers_
```

#### Interval Fuzzy C-Means

```python
from intercluslib.clustering import IntervalFuzzyCMeans

# Create Fuzzy C-Means model
fcm = IntervalFuzzyCMeans(n_clusters=4, m=2.0, max_iter=300)
fcm.fit(data)

print(fcm.U)
```

#### Interval Agglomerative Clustering

```python
from intercluslib.clustering import IntervalAgglomerativeClustering

# Create hierarchical clustering model
agg_clustering = IntervalAgglomerativeClustering(
    n_clusters=4, 
    linkage='ward', 
    distance_metric='euclidean'
)
agg_clustering.fit(data)

print(agg_clustering.labels_)
```

### Preprocessing Tools (`preprocessing` module)

#### Normalization

```python
from intercluslib.preprocessing import MinMaxNormalizer, ZScoreNormalizer

# Min-Max normalization
min_max = MinMaxNormalizer()
data_normalized = min_max.fit_transform(data)

# Z-score normalization
z_score = ZScoreNormalizer()
data_standardized = z_score.fit_transform(data)
```

#### Outlier Detection

```python
from intercluslib.preprocessing import OutlierDetector

# Detect outliers
detector = OutlierDetector(method='iqr', threshold=1.5)
outliers = detector.detect(data)
print(f"Number of outliers detected: {outliers.sum()}")

# Remove outliers
data_clean = detector.remove_outliers(data)
```

#### Imputation

```python
from intercluslib.preprocessing import Imputer

# Impute missing values using mean
imputer = Imputer(strategy='mean')
data_imputed = imputer.fit_transform(data)
```

### Evaluation Methods (`evaluation` module)

```python
from interClusLib.cluster_number_analysis import ElbowMethodEvaluator, GapStatisticEvaluator, LMethodEvaluator

# Determine optimal number of clusters using Elbow Method
elbow = ElbowMethodEvaluator()
optimal_k_elbow = elbow.evaluate(data, k_range=range(2, 10))

# Gap Statistic method
gap = GapStatisticEvaluator(n_bootstraps=50, random_state=42)
optimal_k_gap = gap.evaluate(data, k_range=range(2, 10))

# L Method
l_method = LMethodEvaluator()
optimal_k_l = l_method.evaluate(data, k_range=range(2, 15))

print(f"Optimal k suggested by Elbow Method: {optimal_k_elbow}")
print(f"Optimal k suggested by Gap Statistic: {optimal_k_gap}")
print(f"Optimal k suggested by L Method: {optimal_k_l}")
```

### Metrics and Distance Functions (`metric` module)

```python
from interClusLib.metric import euclidean_distance, manhattan_distance

# Calculate Euclidean distance between two samples
dist_euclidean = euclidean_distance(sample1, sample2)

# Calculate Manhattan distance
dist_manhattan = manhattan_distance(sample1, sample2)
```

### Visualization Tools (`visualization` module)

```python
from interClusLib.visualization import HeatMap, Dendrogram, Interval2d, IntervalRadarChart

# Visualize hierarchical clustering with dendrogram
dendrogram = Dendrogram()
dendrogram.visualize(linkage_matrix, labels=sample_names)

# 2D interval data visualization
plotter_2d = Interval2d()
plotter_2d.visualize(interval_data, labels=labels, features=['temp_min', 'temp_max'])

# Radar chart visualization
radar = IntervalRadarChart()
radar.visualize(cluster_centers, feature_names=feature_names)
```

## Examples

INTERCLUSLIB provides several example Jupyter notebooks to help users get started:

- `example_import.ipynb`: Data import and basic usage
- `example_kmeans.ipynb`: Detailed tutorial on Interval K-Means clustering
- `example_hierarchical.ipynb`: Hierarchical clustering analysis
- `evaluation.ipynb`: Comparison of clustering evaluation methods

Additionally, the `examples` directory contains sample datasets:

- `ChinaTemp.csv`: Temperature interval data for various regions in China
- Other sample data files

## API Documentation

Complete API documentation is available at [https://intercluslib.readthedocs.io/](https://intercluslib.readthedocs.io/).

## Project Structure

```
INTERCLUSLIB/
├── examples/                    # Example notebooks and datasets
├── intercluslib/                # Main package
│   ├── clustering/              # Clustering algorithms
│   ├── evaluation/              # Evaluation methods
│   ├── cluster_number_analysis/ # Evaluation methods
│   ├── metric/                  # Metrics and distance Measures
│   ├── preprocessing/           # Preprocessing tools
│   ├──  visualization/          # Visualization tools
│   └── IntervalData,py          # Interval Data structure
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── setup.py                     # Installation script
├── README.md                    # Project overview
└── LICENSE                      # License file
```

### Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/username/intercluslib.git
cd intercluslib

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_clustering.py

# Generate test coverage report
pytest --cov=intercluslib
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Project Maintainer: [Jiashu CHEN]()
- Project Homepage: [GitHub](https://github.com/hchaenc/InterClusLib)
- Documentation: [ReadTheDocs](https://intercluslib.readthedocs.io/)
- Issue Reporting: [IntervalClusLib Issues](https://github.com/hchaenc/InterClusLib/issues)

## Acknowledgements

We would like to thank all developers and researchers who have contributed to this project. Special thanks to the following open-source projects that have provided significant support for INTERCLUSLIB:

- NumPy
- pandas
- scikit-learn
- matplotlib
- SciPy
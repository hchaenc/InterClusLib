# InterClusLib

InterClusLib is a Python library focused on interval data clustering analysis and visualization. It provides various clustering algorithms, preprocessing tools, evaluation methods, and visualization capabilities.

## Features

- Multiple interval data clustering algorithms (K-Means, Fuzzy C-Means, Hierarchical Clustering, etc.)
- Complete data preprocessing toolkit (normalization, outlier detection, imputation)
- Various clustering evaluation metrics and methods (Elbow Method, Gap Statistic, L Method, etc.)
- Rich visualization tools (Heat Maps, Dendrograms, 2D/3D Interval Plots, Radar Charts, Parallel Coordinates)
- Flexible distance metrics and similarity calculations
- Detailed examples and documentation

## Installation

Install the latest stable version using pip (After uploading to PyPi):

```bash
pip install intercluslib
```

Install the development version from source:

```bash
git clone https://github.com/hchaenc/intercluslib.git
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

Here's a simple example demonstrating how to use InterClusLib for interval data clustering analysis:

```python
from interClusLib.IntervalData import IntervalData
from interClusLib.preprocessing import min_max_normalize
from interClusLib.clustering.IntervalKMeans import IntervalKMeans
from interClusLib.visualization import Interval3d, IntervalParallelCoordinates
from interClusLib.evaluation import *
from interClusLib.cluster_number_analysis import *

# Load example data
temp_data = IntervalData.from_csv("ModifiedChinaTemp.csv")
temp_data.summary()

# Data preprocessing
data = temp_data.get_intervals()
data = min_max_normalize(data)

# Perform clustering using Interval K-Means
kmeans = IntervalKMeans(n_clusters=3, max_iter=1000, distance_func='euclidean', random_state=43)
kmeans.fit(data)

# Visualize evaluation results
fig, ax = Interval3d.visualize(intervals=data, labels=kmeans.labels_, centroids=kmeans.centroids_, margin=0.01)

# Calculate evlauation
metric_results = kmeans.compute_metrics_for_k_range(
    data,
    min_clusters=2,
    max_clusters=10,
    metrics=['distortion', 'silhouette', 'davies_bouldin','calinski_harabasz','dunn'],
)

# Find Optimal Cluster Number
l_method = LMethod(min_clusters=2, max_clusters=10)
optimal_k_l = l_method.evaluate(metric_results['distortion'])
plt = l_method.plot()
print(f"L Method optimal k: {optimal_k_l}")
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

#### Inverted Interval

```python
from interClusLib.preprocessing import InvertedIntervalHandler

# Create handler instance
handler = InvertedIntervalHandler()

# Check for inverted intervals
results = handler.check_intervals(interval_data)
print(f"Found {results['total_inverted']} inverted intervals")

# Fix inverted intervals by swapping bounds
fixed_data = handler.fix_inverted_intervals(interval_data, action='swap')

# Or use convenience method for one-step processing
fixed_data = InvertedIntervalHandler.validate_and_fix(
    interval_data, 
    action='swap',  # Options: 'swap', 'remove', 'set_nan', 'set_equal', 'ignore'
    show_report=True
)
```

#### Missing Data Imputation

```python
from interClusLib.preprocessing import MissingValueImputor

# Create imputor instance
imputor = MissingValueImputor()

# Check for missing values
results = imputor.check_missing_values(interval_data)
print(f"Found {results['total_missing']} missing intervals")

# Impute missing values using different methods
data_imputed = imputor.fix_missing_values(interval_data, action='fill_mean')

# Or use convenience method
data_imputed = MissingValueImputor.validate_and_fix(
    interval_data,
    action='fill_mean',
    show_report=True
)
```

### Evaluation Methods (`evaluation` module)

```python
from interClusLib.cluster_number_analysis import ElbowMethodEvaluator, GapStatisticEvaluator, LMethodEvaluator

# Determine optimal number of clusters using Elbow Method
elbow = ElbowMethod()
optimal_k_elbow = elbow.evaluate(metric_results['distortion'])

# Gap Statistic method
gap = GapStatistic(min_clusters=2, max_clusters=15, n_refs=10)
optimal_k = gap.evaluate(
    eval_data=metric_results['distortion'],
    raw_data=data,
    cluster_func=cluster
)

# L Method
l_method = LMethod(min_clusters=2, max_clusters=10)
optimal_k_l = l_method.evaluate(metric_results['distortion'])

print(f"Optimal k suggested by Elbow Method: {optimal_k_elbow}")
print(f"Optimal k suggested by Gap Statistic: {optimal_k_gap}")
print(f"Optimal k suggested by L Method: {optimal_k_l}")
```

### Distance and Similarity Functions (`metric` module)

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
│   ├── visualization/          # Visualization tools
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
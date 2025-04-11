"""
Interval Clustering Algorithms

This package provides various clustering algorithms designed specifically for
interval-valued data. Interval data is structured as arrays with shape (n_samples, n_dims, 2),
where the last dimension represents the lower and upper bounds of each interval.

Available algorithms:
- IntervalKMeans: K-means clustering adapted for interval data
- IntervalFuzzyCMeans: Fuzzy C-means clustering for interval data
- IntervalAgglomerativeClustering: Agglomerative hierarchical clustering for interval data
"""

from .IntervalKMeans import IntervalKMeans
from .IntervalFuzzyCMeans import IntervalFuzzyCMeans
from .IntervalAgglomerativeClustering import IntervalAgglomerativeClustering
from .AbstractIntervalClustering import AbstractIntervalClustering

__all__ = [
    'IntervalKMeans',
    'IntervalFuzzyCMeans',
    'IntervalAgglomerativeClustering',
    'AbstractIntervalClustering'
]
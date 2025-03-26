"""
A package for determining the optimal number of clusters in clustering algorithms.

This package provides implementations of several methods for finding the
optimal number of clusters, including the Elbow Method, L Method, Gap Statistic,
and Information Criterion approaches.
"""

# Import main classes for easier access
from .base_evaluator import ClusterEvaluationMethod
from .elbow_method import ElbowMethod
from .l_method import LMethod
from .gap_statistic import GapStatistic
from .max_min_evaluator import MaxMinClusterEvaluator

# Define version
__version__ = '0.1.0'

# Define what is available through "from package import *"
__all__ = [
    'ClusterEvaluationMethod',
    'ElbowMethod',
    'LMethod',
    'GapStatistic',
    'MaxMinClusterEvaluator',
]
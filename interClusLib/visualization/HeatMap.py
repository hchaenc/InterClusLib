import pandas as pd
import numpy as np
import os
import seaborn as sns

class HeatMap:

    def draw_heat_map(distance_matrix, ax = None ,cmap = 'coolwarm', annot=False, cbar=True):
        sns.heatmap(distance_matrix, cmap = cmap, annot = annot, cbar = cbar, ax = ax )
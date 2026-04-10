# DATA EXPLORATION FUNCTIONS

# Imports
import numpy as np


def summarize(name, data, unit=""):
    data = np.array(data)
    print(f"\n=== {name} ===")
    print(f"Min  : {data.min():.1f}{unit}")
    print(f"Max  : {data.max():.1f}{unit}")
    print(f"Mean : {data.mean():.1f}{unit}")
    print(f"Std  : {data.std():.1f}{unit}")
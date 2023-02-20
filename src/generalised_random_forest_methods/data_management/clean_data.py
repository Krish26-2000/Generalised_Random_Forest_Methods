"""Function(s) for saving the data set(s)."""

import numpy as np
import pandas as pd
from econml.grf import CausalForest, CausalIVForest, RegressionForest
from econml.dml import CausalForestDML
import scipy.special
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def ipums_data(url_1):
    """Import and save the data created from IPUMS.

    Args:
        url(str): The url contains the dataset

    Returns:
        data(pandas.DataFrame): The IPUMS data.
    
    """
    data = pd.read_csv(url_1)
    return data

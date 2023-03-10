import numpy as np
import pandas as pd
from econml.grf import CausalForest, CausalIVForest, RegressionForest
from econml.dml import CausalForestDML
import scipy.special
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.linear_model import LassoCV



def train_causal_forest_model():
    """

    Returns:

    """
    # set parameters for causal forest
    causal_forest = CausalForestDML(criterion='het',
                                    n_estimators=10000,
                                    min_samples_leaf=10,
                                    max_depth=None,
                                    max_samples=0.5,
                                    discrete_treatment=False,
                                    honest=True,
                                    inference=True,
                                    cv=10,
                                    model_t=LassoCV(),
                                    model_y=LassoCV(),
                                    )
    return causal_forest




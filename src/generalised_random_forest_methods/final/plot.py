import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from generalised_random_forest_methods.config import BLD


def feature_importance_plot(x, y):

    fig, ax = plt.subplots()
    sns.barplot(x=x, y=y, color='C0').set(title='Feature Importances', ylabel='Importance')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right");
    return fig


def treatment_effect_plot(z):
    """Creating plot depicting the treatment effects and the 
        confidence intervals

    Args:
        z: data containing rolling means of the treatment 
            effects and confidence intervals

    Returns:
        fig: A plot depicting the treatment effects and the 
            confidence effects determined by the causal forest 
            model
    
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    # plotlines for treatment effects and confidence intervals
    ax.plot(
        z["cate"],
        marker=".",
        linestyle="-",
        linewidth=0.5,
        label="CATE",
        color="indigo",
    )
    ax.plot(z["lb"], marker=".", linestyle="-", linewidth=0.5, color="steelblue")
    ax.plot(z["ub"], marker=".", linestyle="-", linewidth=0.5, color="steelblue")
    # axes and create legend
    ax.set_ylabel("Treatment Effects")
    ax.set_xlabel("Number of observations")
    ax.set_ylim([-2, 2])
    ax.legend()
    return fig

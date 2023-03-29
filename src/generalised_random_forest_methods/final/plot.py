import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def feature_importance_plot(x, y):
    """Creating plot depicting the significance of each variable
    on the outcome variable

    Args:
        x: data containing the columns of X variable
        y: data containing the outcome of feature_imporatance from the 
            model

    Returns:
        fig: A plot which explains the influence of each variable on the 
            outcome.
    
    """
    fig, ax = plt.subplots()
    sns.barplot(x=x, y=y, color='C0').set(
    title='Feature Importances', ylabel='Importance')
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

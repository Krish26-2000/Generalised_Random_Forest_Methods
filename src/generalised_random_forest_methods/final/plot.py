import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from generalised_random_forest_methods.config import BLD

path = BLD / "python" / "data" / "z_rolling_means.pkl"


def treatment_effect_plot(z):
    z = pd.read_pickle(path)
    fig, ax = plt.subplots(figsize =(12, 8))
    # plotlines for treatment effects and confidence intervals
    ax.plot(z['cate'],
            marker='.', linestyle='-', linewidth=0.5, label='CATE', color='indigo')
    ax.plot(z['lb'],
            marker='.', linestyle='-', linewidth=0.5, color='steelblue')
    ax.plot(z['ub'],
            marker='.', linestyle='-', linewidth=0.5, color='steelblue')
    # axes and create legend
    ax.set_ylabel('Treatment Effects')
    ax.set_xlabel('Number of observations')
    ax.legend()
    return fig


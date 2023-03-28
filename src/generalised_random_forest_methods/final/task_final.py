import pandas as pd
import pytask
from generalised_random_forest_methods.config import BLD, SRC
from generalised_random_forest_methods.final.plot import treatment_effect_plot
from generalised_random_forest_methods.final.plot import treatment_effect_plot2
from generalised_random_forest_methods.final.plot import feature_importance_plot 
from generalised_random_forest_methods.utilities import read_yaml
from generalised_random_forest_methods.analysis.model import train_causal_forest_model
from generalised_random_forest_methods.analysis.model import return_param_dict

@pytask.mark.depends_on(
        {
           "data": BLD / "python" / "data" / "trained_df.pkl",
           "data_info": SRC / "data_management" / "data_info.yaml",
        }
    )
@pytask.mark.produces(BLD / "python" / "plots" / "feature_importance_plot.png")
def task_feature_importance(depends_on, produces):

    data = pd.read_pickle(depends_on["data"])
    data_info = read_yaml(depends_on["data_info"])

    X = train[data_info["features"]]
    params = return_param_dict()
    features_plot = feature_importance_plot(x= X.columns, y= train_causal_forest_model(params).feature_importances([0]))
    features_plot.savefig(produces)


@pytask.mark.depends_on(BLD / "python" / "data" / "z_rolling_means.pkl")
@pytask.mark.produces(BLD / "python" / "plots" / "treatment_effects_plot.png")
def task_plot_treatment_effects(depends_on, produces):
    """Save the treatment effect plot 

    Args:
        depends_on(str): the rolling means dataframe
        produces(str): the folder path containing data to be stored

    Returns:
        The treatment effects plot
    
    """
    z = pd.read_pickle(depends_on)
    plot = treatment_effect_plot(z)
    plot.savefig(produces)


@pytask.mark.depends_on(BLD / "python" / "data" / "z_rolling_means2.pkl")
@pytask.mark.produces(BLD / "python" / "plots" / "children_treatment_effects_plot.png")
def task_plot_treatment_effects2(depends_on, produces):
    """Save the treatment effect plot for the 2nd hypothesis

    Args:
        depends_on(str): the rolling_means2 dataframe
        produces(str): the folder path containing data to be stored

    Returns:
        The children treatment effects plot
    
    """
    z2 = pd.read_pickle(depends_on)
    plot = treatment_effect_plot2(z2)
    plot.savefig(produces)

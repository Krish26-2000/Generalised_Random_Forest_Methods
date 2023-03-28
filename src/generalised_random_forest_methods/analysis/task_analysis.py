import pickle

import pandas as pd
import pytask
from generalised_random_forest_methods.analysis.model import return_child_param_dict
from generalised_random_forest_methods.analysis.model import return_param_dict
from generalised_random_forest_methods.analysis.model import train_causal_forest_model
from generalised_random_forest_methods.config import BLD
from generalised_random_forest_methods.config import SRC
from generalised_random_forest_methods.utilities import read_yaml


@pytask.mark.depends_on(
    {
        "train": BLD / "python" / "data" / "train_df.pkl",
        "test": BLD / "python" / "data" / "test_df.pkl",
        "data_info": SRC / "data_management" / "data_info.yaml",
    }
)
@pytask.mark.produces(BLD / "python" / "data" / "treatment_effects_dict.pkl")
def task_fit_causal_forest(depends_on, produces):
    """Fit causal forest model on the data variables and get the treatment effects

    Args:
        depends_on(str):
            - training data from final_df
            - testing data from final_df
            - data_info.yaml which contains information of variables in study
        produces(str): the folder path containing data to e stored

    Returns:
        treatment_effects_dict(pickle): pickle file containing data on treatmnet effects and
                                        confidence intervals

    """
    train = pd.read_pickle(depends_on["train"])
    test = pd.read_pickle(depends_on["test"])
    data_info = read_yaml(depends_on["data_info"])

    Y = train[data_info["outcome"]]
    X = train[data_info["features"]]
    T = train[data_info["treatment"]]
    W = train[data_info["instrument"]]
    X_test = test[data_info["features"]]

    Y_array = Y.to_numpy().reshape(-1, 1)
    T_array = T.to_numpy().reshape(-1, 1)
    X_array = X.to_numpy()
    W_array = W.to_numpy().reshape(-1, 1)

    params = return_param_dict()
    model = train_causal_forest_model(params).fit(
        Y_array, T_array, X=X_array, W=W_array
    )
    CATE = model.const_marginal_ate(X_test)
    print(f"CATE: {CATE}")
    treatment_effects = model.effect(X)
    fea_impo = model.feature_importances()[0]
    # calculate lower bound and upper bound confidence intervals
    lb, ub = model.effect_interval(X, alpha=0.05)
    causal_forest_data = {"treatment_effects": treatment_effects, "lb": lb, "ub": ub,
                          "feature_importance": fea_impo}
    with open(produces, "wb") as file:
        pickle.dump(causal_forest_data, file)


@pytask.mark.depends_on(BLD / "python" / "data" / "treatment_effects_dict.pkl")
@pytask.mark.produces(BLD / "python" / "data" / "df.pkl")
def task_get_effects(depends_on, produces):
    """Create dataframes of treatment effects and confidence intervals 
        for creating a plot

    Args:
        depends_on(str): The treatment effects dictionary created in the 
                    previous task 
        produces(str): the folder path containing data to be stored

    Returns:
        df(pickle): A dataframe which contains treatment effects and 
                    confidence intervals

    """
    with open(depends_on, "rb") as file:
        causal_forest_data = pickle.load(file)

    # convert arrays to pandas dataframes for plotting
    te_df = pd.DataFrame(causal_forest_data["treatment_effects"], columns=["cate"])
    lb_df = pd.DataFrame(causal_forest_data["lb"], columns=["lb"])
    ub_df = pd.DataFrame(causal_forest_data["ub"], columns=["ub"])

    # merge dataframes and sort
    df = te_df.merge(lb_df, left_index=True, right_index=True, how="left")
    df = df.merge(ub_df, left_index=True, right_index=True, how="left")
    df.sort_values("cate", inplace=True, ascending=True)
    df.reset_index(inplace=True, drop=True)
    df.to_pickle(produces)


@pytask.mark.depends_on(BLD / "python" / "data" / "df.pkl")
@pytask.mark.produces(BLD / "python" / "data" / "z_rolling_means.pkl")
def task_calculate_z_means(depends_on, produces):
    """Calculate rolling means

    Args:
        depends_on(str): df.pkl which is dataframe created in the 
                        previous task
        produces(str): the folder path containing data to be stored

    Returns:
        z_rolling_means(pickle): a dataframe which contains rolling means
                                for creating the graph

    """
    df = pd.read_pickle(depends_on)
    # calculate rolling mean
    z = df.rolling(window=30, center=True).mean()
    with open(produces, "wb") as file:
        pickle.dump(z, file)


# Fit the same model with the child hypothesis data
@pytask.mark.depends_on(
    {
        "train": BLD / "python" / "data" / "train2_df.pkl",
        "test": BLD / "python" / "data" / "test2_df.pkl",
        "data_info": SRC / "data_management" / "data_info.yaml",
    }
)
@pytask.mark.produces(BLD / "python" / "data" / "treatment_effects_dict2.pkl")
def task_fit_causal_forest2(depends_on, produces):
    """Fit causal forest model for the children hypothesis on the data variables and 
        get the treatment effects

    Args:
        depends_on(str):
            - training data from final_df for 2nd hypothesis
            - testing data from final_df for 2nd hypothesis
            - data_info.yaml which contains information of variables in study
        produces(str): the folder path containing data to e stored

    Returns:
        treatment_effects_dict2(pickle): pickle file containing data on treatmnet effects and
                                        confidence intervals

    """
    train = pd.read_pickle(depends_on["train"])
    test = pd.read_pickle(depends_on["test"])
    data_info = read_yaml(depends_on["data_info"])

    Y2 = train[data_info["outcome2"]]
    X2 = train[data_info["features2"]]
    T2 = train[data_info["treatment"]]
    W2 = train[data_info["instrument"]]
    X2_test = test[data_info["features2"]]

    Y_array = Y2.to_numpy().reshape(-1, 1)
    T_array = T2.to_numpy().reshape(-1, 1)
    X_array = X2.to_numpy()
    W_array = W2.to_numpy().reshape(-1, 1)

    params_child = return_child_param_dict()
    model = train_causal_forest_model(params_child).fit(
        Y_array, T_array, X=X_array, W=W_array
    )
    CATE = model.const_marginal_ate(X2_test)
    print(f"CATE: {CATE}")
    treatment_effects = model.effect(X2)
    fea_impo = model.feature_importances()[0]
    # calculate lower bound and upper bound confidence intervals
    lb, ub = model.effect_interval(X2, alpha=0.05)
    causal_forest_data = {"treatment_effects": treatment_effects, "lb": lb, "ub": ub,
                          "feature_importance": fea_impo}
    with open(produces, "wb") as file:
        pickle.dump(causal_forest_data, file)


@pytask.mark.depends_on(BLD / "python" / "data" / "treatment_effects_dict2.pkl")
@pytask.mark.produces(BLD / "python" / "data" / "df2.pkl")
def task_get_effects2(depends_on, produces):
    """Create dataframes of treatment effects and confidence intervals 
        for creating a plot for the 2nd hypothesis

    Args:
        depends_on(str): The treatment effects dictionary created in the 
                    previous task 
        produces(str): the folder path containing data to be stored

    Returns:
        df2(pickle): A dataframe which contains treatment effects and 
                    confidence intervals for this hypothesis

    """
    with open(depends_on, "rb") as file:
        causal_forest_data = pickle.load(file)

    # convert arrays to pandas dataframes for plotting
    te_df = pd.DataFrame(causal_forest_data["treatment_effects"], columns=["cate"])
    lb_df = pd.DataFrame(causal_forest_data["lb"], columns=["lb"])
    ub_df = pd.DataFrame(causal_forest_data["ub"], columns=["ub"])

    # merge dataframes and sort
    df2 = te_df.merge(lb_df, left_index=True, right_index=True, how="left")
    df2 = df2.merge(ub_df, left_index=True, right_index=True, how="left")
    df2.sort_values("cate", inplace=True, ascending=True)
    df2.reset_index(inplace=True, drop=True)
    df2.to_pickle(produces)


@pytask.mark.depends_on(BLD / "python" / "data" / "df2.pkl")
@pytask.mark.produces(BLD / "python" / "data" / "z_rolling_means2.pkl")
def task_calculate_z_means2(depends_on, produces):
    """Calculate rolling means for creating the plot for 2nd hypothesis

    Args:
        depends_on(str): df.pkl which is dataframe created in the 
                        previous task
        produces(str): the folder path containing data to be stored

    Returns:
        z_rolling_means(pickle): a dataframe which contains rolling means
                                for creating the graph

    """
    df = pd.read_pickle(depends_on)
    # calculate rolling mean
    z2 = df.rolling(window=30, center=True).mean()
    with open(produces, "wb") as file:
        pickle.dump(z2, file)

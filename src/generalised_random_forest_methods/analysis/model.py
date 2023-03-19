from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV


def train_causal_forest_model(param_dict):
    """
    Args:
       param_dict(dict): The dictionary containing parameters of the model

    Returns:
       causal_forest: the analysis from model from econml library

    """
    # set parameters for causal forest
    causal_forest = CausalForestDML(**param_dict)
    return causal_forest


def return_param_dict():
    return {
        "criterion": "het",
        "n_estimators": 10000,
        "min_samples_leaf": 10,
        "max_depth": None,
        "max_samples": 0.5,
        "discrete_treatment": False,
        "honest": True,
        "inference": True,
        "cv": 10,
        "model_t": LassoCV(),
        "model_y": LassoCV(),
    }


def return_child_param_dict():
    return {
        "criterion": "het",
        "n_estimators": 20000,
        "min_samples_leaf": 20,
        "max_depth": None,
        "max_samples": 0.5,
        "discrete_treatment": False,
        "honest": True,
        "inference": True,
        "cv": 10,
        "model_t": LassoCV(),
        "model_y": LassoCV(),
    }

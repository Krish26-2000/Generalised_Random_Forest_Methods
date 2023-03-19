"""Function(s) for saving the data set(s)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special
from dowhy import CausalModel
from econml.dml import CausalForestDML
from econml.grf import CausalForest
from econml.grf import CausalIVForest
from econml.grf import RegressionForest
from IPython.display import display
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree


def ipums_data(url):
    """Import and save the data created from IPUMS.

    Args:
        url(str): The url contains the dataset

    Returns:
        data(pandas.DataFrame): The IPUMS data.

    """
    data = pd.read_csv(url)
    return data


def clean_data(data, data_info):
    """Cleaning the IPUMS dataset.

    Information on columns to drop is stored in ``data_management\data_info.yaml``

    Args:
        data (pandas.DataFrame): The IPUMS data set.
            - 'MARRINYR'(int): column of the data which is limited to 2 indicating the couples
                                married in the previous year
            - 'SEX_SP'(int): column dtype changed to int.
            - 'AGE'(int): column enteries stored which are greater than 18
            - 'MARST'(int): column of the data limited to 1 to include only married couples
            - 'SEX_SP', 'MARST'(int): columns to be renamed
            - 'ELDCH'(int): age of elder child column cleaned

        data_info (dict): Information on data set stored in data_info.yaml. The
            following keys can be accessed:
            - 'columns_to_drop'(list): Names of variable column in the data to drop

    Returns:
        data(pandas.DataFrame): The cleaned_df DataFrame.

    """
    columns_to_drop = data_info["columns_to_drop"]

    data = data.drop(columns=columns_to_drop)
    data = data.dropna()
    data = data[data["MARRINYR"] == 2]
    data["SEX_SP"] = data["SEX_SP"].astype(int)
    data = data[data["AGE"] > 18]
    data = data[data["MARST"] == 1]
    data = data.rename(columns={"SEX_SP": "spouse_sex", "MARST": "married_status"})
    data.loc[data.ELDCH == 99, "ELDCH"] = 0
    return data


def create_columns(data):
    """Create new variables for analysis.

    Args:conda
        data (pandas.DataFrame): The cleaned_df DataFrame
            - 'same_sex_couple'(int): variable where the sex of the person and the spouse is same
            - 'married_year'(int): column of year of marriage
            - 'child_birth_year'(int): column created by subtracting the child's age from 2023
            - 'outcome'(int): variable which indicate whether couple married after 2015
            - 'outcome_child'(int): variable which indicate whether couple started a family after 2015

    Returns:
        data(pandas.DataFrame): the final_df DataFrame

    """
    data.loc[:, "same_sex_couple"] = 0
    data.loc[data["SEX"] == data["spouse_sex"], "same_sex_couple"] = 1
    data.loc[:, "married_year"] = data["YEAR"] - 1
    data.loc[:, "child_birth_year"] = 2023 - data["ELDCH"]
    data["outcome"] = (data["married_year"] > 2015).astype(int)  # Y
    data["outcome_child"] = (data["child_birth_year"] > 2015).astype(int)  # Y2
    return data


def causal_model(data, data_info):
    """Creates a visual model to understand the model.

    Args:
        data(pandas.DataFrame): the final_df dataset
        data_info(dict): Information on data set stored in data_info.yaml. The
            following keys can be accessed:
            - 'treatment': T_i or binary treatment variable
            - 'outcome': Y_i or dependent variable which is binary
            - 'features': X_i or independent variables
            - 'instrument': W_i or instrument variable


    Returns:
        model: a model which can be viewed and saved in png format

    """
    model = CausalModel(
        data=data,
        treatment=data_info["treatment"],
        outcome=data_info["outcome"],
        common_causes=data_info["features"],
        instruments=data_info["instrument"],
        effect_modifiers=None,
    )
    return model


def train_test_data(data, test_size=0.2):
    """Split the dataset into training and testing datasets.

    Args:
        data(pandas.DataFrame): The final_df dataset

    Returns:
        train(pandas.DataFrame): The training dataset
        test(pandas.DataFrame):The testing dataset

    """
    train, test = train_test_split(data, test_size=test_size)
    return train, test


def causal_model_child(data, data_info):
    """Creates a visual model to understand the children causal model.

    Args:
        data(pandas.DataFrame): the final_df dataset
        data_info(dict): Information on data set stored in data_info.yaml. The
            following keys can be accessed:
            - 'treatment': T_i or binary treatment variable
            - 'outcome2': Y_i or dependent variable which is binary
            - 'features2': X_i or independent variables
            - 'instrument': W_i or instrument variable


    Returns:
        model: a model which can be viewed and saved in png format

    """
    model = CausalModel(
        data=data,
        treatment=data_info["treatment"],
        outcome=data_info["outcome2"],
        common_causes=data_info["features2"],
        instruments=data_info["instrument"],
        effect_modifiers=None,
    )
    return model

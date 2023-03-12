"""Function(s) for saving the data set(s)."""

import numpy as np
import pandas as pd
from econml.grf import CausalForest, CausalIVForest, RegressionForest
from econml.dml import CausalForestDML
import scipy.special
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from dowhy import CausalModel
from sklearn.model_selection import train_test_split
from IPython.display import Image,display

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
    """

    Args:
        data:

    Returns:

    """
    columns_to_drop = data_info["columns_to_drop"]

    data = data.drop(columns=columns_to_drop)
    data = data.dropna()
    data = data[data["MARRINYR"] == 2] 
    data['SEX_SP'] = data['SEX_SP'].astype(int)
    data = data[data["AGE"] > 18] 
    data = data[data["MARST"] == 1]
    data = data.rename(columns = {'SEX_SP' : 'spouse_sex', 'MARST' : 'married_status'})
    return data


def create_columns(data):
    """
    
    Args:conda
        data:

    Returns:

    """
    data.loc[:, 'same_sex_couple'] = 0
    data.loc[data['SEX'] == data['spouse_sex'], 'same_sex_couple'] = 1
    data.loc[:, 'married_year'] = data["YEAR"] - 1
    data["married_year"].value_counts() #to understand how many people
                                        #married in each year.
    data["outcome"] = (data["married_year"] > 2015).astype(int)  # Y
    return data


def return_var_dict():
    """

    Args:
        data:

    Returns:

    """
    var_dict = {
                'features': ['SEX', 'RACE', 'EDUC', 'EMPSTAT', 'AGE'],  # X
                'treatment': 'same_sex_couple',  # T
                'instrument': 'INCTOT',  # W
                'outcome': 'outcome',  # Y
                    }
    return var_dict


def causal_model(data, features, treatment, instrument, outcome):
    """

    Args:
        features:
        treatment:
        instrument:
        outcome:

    Returns:

    """
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        common_causes=features,
        instruments=instrument,
        effect_modifiers=None )
    return model


def train_test_data(data):
    """

    Args:
        data:

    Returns:

    """
    train, test = train_test_split(data, test_size=0.2)
    return train, test



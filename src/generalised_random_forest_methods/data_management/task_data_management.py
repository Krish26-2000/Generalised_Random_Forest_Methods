import pandas as pd
import pytask
from generalised_random_forest_methods.config import BLD
from generalised_random_forest_methods.config import SRC
from generalised_random_forest_methods.data_management.clean_data import causal_model
from generalised_random_forest_methods.data_management.clean_data import (
    causal_model_child,
)
from generalised_random_forest_methods.data_management.clean_data import clean_data
from generalised_random_forest_methods.data_management.clean_data import create_columns
from generalised_random_forest_methods.data_management.clean_data import ipums_data
from generalised_random_forest_methods.data_management.clean_data import train_test_data
from generalised_random_forest_methods.utilities import read_yaml

url = "https://www.dropbox.com/s/s2gr0x6wy1ms3e6/usa_00007.csv?dl=1"


@pytask.mark.produces(BLD / "python" / "usa_00007.csv")
def task_save_ipums_data(produces):
    """Import IPUMS data from Dropbox folder and save in the BLD folder.

    Args:
        produces: Specify the folder path to locate the file

    Returns:
        data(pandas.DataFrame): The IPUMS dataset

    """
    data = ipums_data(url)
    data.to_csv(produces)


@pytask.mark.depends_on(
    {
        "ipums_data": BLD / "python" / "usa_00007.csv",
        "data_info": SRC / "data_management" / "data_info.yaml",
    }
)
@pytask.mark.produces(BLD / "python" / "data" / "clean_df.pkl")
def task_clean_df(depends_on, produces):
    """Reads the data, cleans it with the function and saves the cleaned data.

    Args:
        depends_on(str):
            - The usa_00007.csv data file
            - the dictionary called data_info.yaml
        produces(str): the folder path containing data to be stored

    Returns:
        cleaned_df(pickle): the cleaned datafile

    """
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["ipums_data"])
    clean_df = clean_data(data, data_info)
    clean_df.to_pickle(produces)


@pytask.mark.depends_on(BLD / "python" / "data" / "clean_df.pkl")
@pytask.mark.produces(BLD / "python" / "data" / "final_df.pkl")
def task_create_variables(depends_on, produces):
    """Add new created variables to the clean_df and save it to make final_df.

    Args:
        depends_on(str): The cleaned_df
        produces(str): the folder path containing data to be stored

    Returns:
        final_df(pickle): the final datafile

    """
    clean_df = pd.read_pickle(depends_on)
    final_df = create_columns(clean_df)
    final_df.to_pickle(produces)


@pytask.mark.depends_on(
    {
        "final_df": BLD / "python" / "data" / "final_df.pkl",
        "data_info": SRC / "data_management" / "data_info.yaml",
    }
)
@pytask.mark.produces(BLD / "python" / "plots" / "causal_model.png")
def task_casual_model(depends_on, produces):
    """Creates the causal model and saves in png format.

    Args:
        depends_on(str):
            - the final_df dataset
            - data_info.yaml which has information about the variables
        produces(str): the folder path containing data to be stored


    Returns:
        model(png): causal_model image

    """
    data = pd.read_pickle(depends_on["final_df"])
    data_info = read_yaml(depends_on["data_info"])
    model = causal_model(data, data_info)
    model.view_model(file_name=str(produces)[:-4])


@pytask.mark.depends_on(BLD / "python" / "data" / "final_df.pkl")
@pytask.mark.produces(
    [BLD / "python" / "data" / "train_df.pkl", BLD / "python" / "data" / "test_df.pkl"]
)
def task_train_test_data(depends_on, produces):
    """Splits the data into training and testing data.

    Args:
        depends_on(str): the final_df dataset
        produces(str): the folder path containing data to be stored

    Returns:
        df(pickle): the training and testing dataframes

    """
    data = pd.read_pickle(depends_on)
    train_test_df = train_test_data(data)
    for i, df in enumerate(train_test_df):
        df.to_pickle(produces[i])


@pytask.mark.depends_on(
    {
        "final_df": BLD / "python" / "data" / "final_df.pkl",
        "data_info": SRC / "data_management" / "data_info.yaml",
    }
)
@pytask.mark.produces(BLD / "python" / "plots" / "child_causal_model.png")
def task_child_casual_model(depends_on, produces):
    """Creates the causal model and saves in png format for child hypothesis.

    Args:
        depends_on(str):
            - the final_df dataset
            - data_info.yaml which has information about the variables
        produces(str): the folder path containing data to be stored


    Returns:
        model(png): child_causal_model image

    """
    data = pd.read_pickle(depends_on["final_df"])
    data_info = read_yaml(depends_on["data_info"])
    model = causal_model_child(data, data_info)
    model.view_model(file_name=str(produces)[:-4])


@pytask.mark.depends_on(BLD / "python" / "data" / "final_df.pkl")
@pytask.mark.produces(
    [
        BLD / "python" / "data" / "train2_df.pkl",
        BLD / "python" / "data" / "test2_df.pkl",
    ]
)
def task_train_test_data2(depends_on, produces):
    """Splits the data into training and testing data for child hypothesis.

    Args:
        depends_on(str): the final_df dataset
        produces(str): the folder path containing data to be stored

    Returns:
        df(pickle): the training and testing dataframes

    """
    data = pd.read_pickle(depends_on)
    train_test_df = train_test_data(data, test_size=0.3)
    for i, df in enumerate(train_test_df):
        df.to_pickle(produces[i])

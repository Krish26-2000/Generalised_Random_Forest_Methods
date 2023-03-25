import numpy as np
import pandas as pd
import pytest
from generalised_random_forest_methods.config import TEST_DIR
from generalised_random_forest_methods.data_management.clean_data import clean_data, create_columns
from generalised_random_forest_methods.utilities import read_yaml


@pytest.fixture()
def data():
    # Read the data_fixture.csv file into a dataframe called data
    data = pd.read_csv(TEST_DIR / "data_management" / "data_fixture.csv")
    return data


@pytest.fixture()
def data_info():
    # Read the data_info_fixture.yaml file into a dataframe called data_info
    data_info = read_yaml(TEST_DIR / "data_management" / "data_info_fixture.yaml")
    return data_info


def test_clean_data_drop_columns(data, data_info):
    # clean_data function from clean_data.py drops the columns specified in data_info
    data_clean = clean_data(data, data_info)

    # checks whether the columns in data_info are not there in data_clean 
    assert not set(data_info["columns_to_drop"]).intersection(set(data_clean.columns))


def test_clean_data_dropna(data,data_info):
    # clean_data function from clean_data.py is used to drop NaN values
    data_clean = clean_data(data,data_info)

    #checks if there are any NaN values in data_clean
    assert not data_clean.isnull().any(axis=None)


def test_married_year(data,data_info):
    # clean_data function from clean_data.py has an input that it only keeps the observations 
    # in MARRINYR which is 2 (only married couples are stored in the data)
    data_married_year_cleaned = clean_data(data,data_info)

    # checks if the column has only the entry of 2
    assert data_married_year_cleaned["MARRINYR"].isin([2]).all()
    assert "spouse_sex" in list(data_married_year_cleaned.columns), "'spouse_sex' column does not exist in the df, choose a different new_col_name"

 


def test_all_new_columns(data,data_info):
    cleaned_data = clean_data(data, data_info)
    new_columns_data = create_columns(cleaned_data)

    assert all(new_columns_data["same_sex_couple"].isin([0,1]))

    assert all(new_columns_data["married_year"] == new_columns_data["YEAR"] - 1)

    assert new_columns_data["outcome"].dtype == int 
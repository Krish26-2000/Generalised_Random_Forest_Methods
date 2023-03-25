import numpy as np
import pandas as pd
import pytest
from generalised_random_forest_methods.config import TEST_DIR
from generalised_random_forest_methods.data_management.clean_data import clean_data
from generalised_random_forest_methods.utilities import read_yaml


@pytest.fixture()
def data():
    data = pd.read_csv(TEST_DIR / "data_management" / "data_fixture.csv")
    return data


@pytest.fixture()
def data_info():
    data_info = read_yaml(TEST_DIR / "data_management" / "data_info_fixture.yaml")
    return data_info


def test_clean_data_drop_columns(data, data_info):
    data_clean = clean_data(data, data_info)
    assert not set(data_info["columns_to_drop"]).intersection(set(data_clean.columns))


def test_clean_data_dropna(data,data_info):
    data_clean = clean_data(data,data_info)
    assert not data_clean.isnull().any(axis=None)


def test_married_year(data,data_info):
    data_married_year_cleaned = clean_data(data,data_info)
    assert data_married_year_cleaned["MARRINYR"].isin([2]).all()
import numpy as np
import pandas as pd
import pytest
from generalised_random_forest_methods.analysis.model import fit_logit_model


@pytest.fixture()
def data():
    np.random.seed(0)
    x = np.random.normal(size=100_000)
    coef = 2.0
    prob = 1 / (1 + np.exp(-coef * x))
    y = np.random.binomial(1, prob)
    return pd.DataFrame({"outcome_numerical": y, "covariate": x})


@pytest.fixture()
def data_info():
    return {"outcome": "outcome", "outcome_numerical": "outcome_numerical"}


def test_fit_logit_model_recover_coefficients(data, data_info):
    model = fit_logit_model(data, data_info, model_type="linear")
    params = model.params
    assert np.abs(params["Intercept"]) < 10e-2
    assert np.abs(params["covariate"] - 2.0) < 10e-2


def test_fit_logit_model_error_model_type(data, data_info):
    with pytest.raises(ValueError):  # noqa: PT011
        assert fit_logit_model(data, data_info, model_type="quadratic")

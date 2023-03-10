import pytask
from generalised_random_forest_methods.config import BLD

import pandas as pd
from generalised_random_forest_methods.analysis.model import train_causal_forest_model
import pickle
from econml._cate_estimator import BaseCateEstimator

@pytask.mark.depends_on(
    {
        "Y": BLD / "python" / "data" / "outcome_train.pkl",
        "X": BLD / "python" / "data" / "features_train.pkl",
        "T": BLD / "python" / "data" / "treatment_train.pkl",
        "W": BLD / "python" / "data" / "instrument_train.pkl",
        "X_test": BLD / "python" / "data" / "features_test.pkl",
    }
)
@pytask.mark.produces(BLD / "python" / "data" / "const_marginal_ATE.pkl")
def task_fit_causal_forest(depends_on, produces):
    Y = pd.read_pickle(depends_on["Y"])
    X = pd.read_pickle(depends_on["X"])
    T = pd.read_pickle(depends_on["T"])
    W = pd.read_pickle(depends_on["W"])
    X_test = pd.read_pickle(depends_on["X_test"])

    Y_array = Y.to_numpy().reshape(-1, 1)
    T_array = T.to_numpy().reshape(-1, 1)
    X_array = X.to_numpy()
    W_array = W.to_numpy().reshape(-1, 1)

    model = train_causal_forest_model().fit(Y_array, T_array, X=X_array, W=W_array)
    CATE = model.const_marginal_ate(X_test)
    print(f"CATE: {CATE}")
    with open(produces, 'wb') as file:
        pickle.dump(CATE, file)



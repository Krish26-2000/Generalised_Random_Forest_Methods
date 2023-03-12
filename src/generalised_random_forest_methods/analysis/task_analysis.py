import pytask
from generalised_random_forest_methods.config import BLD

import pandas as pd
from generalised_random_forest_methods.analysis.model import train_causal_forest_model
import pickle


@pytask.mark.depends_on(
    {
        "Y": BLD / "python" / "data" / "outcome_train.pkl",
        "X": BLD / "python" / "data" / "features_train.pkl",
        "T": BLD / "python" / "data" / "treatment_train.pkl",
        "W": BLD / "python" / "data" / "instrument_train.pkl",
        "X_test": BLD / "python" / "data" / "features_test.pkl",
    }
)
@pytask.mark.produces(BLD / "python" / "data" / "treatment_effects_dict.pkl")
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
    treatment_effects = model.effect(X)
    # calculate lower bound and upper bound confidence intervals
    lb, ub = model.effect_interval(X, alpha=0.05)
    causal_forest_data = {
        'treatment_effects': treatment_effects,
        'lb': lb,
        'ub': ub
    }
    with open(produces, 'wb') as file:
        pickle.dump(causal_forest_data, file)


@pytask.mark.depends_on(BLD / "python" / "data" / "treatment_effects_dict.pkl")
@pytask.mark.produces(BLD / "python" / "data" / "df.pkl")
def task_get_effects(depends_on,produces):
    with open(depends_on, 'rb') as file:
        causal_forest_data = pickle.load(file)

    # convert arrays to pandas dataframes for plotting
    te_df = pd.DataFrame(causal_forest_data['treatment_effects'], columns=['cate'])
    lb_df = pd.DataFrame(causal_forest_data['lb'], columns=['lb'])
    ub_df = pd.DataFrame(causal_forest_data['ub'], columns=['ub'])

    # merge dataframes and sort
    df = te_df.merge(lb_df, left_index=True, right_index=True, how='left')
    df = df.merge(ub_df, left_index=True, right_index=True, how='left')
    df.sort_values('cate', inplace=True, ascending=True)
    df.reset_index(inplace=True, drop=True)
    df.to_pickle(produces)


@pytask.mark.depends_on(BLD / "python" / "data" / "df.pkl")
@pytask.mark.produces(BLD / "python" / "data" / "z_rolling_means.pkl")
def task_calculate_z_means(depends_on, produces):
    df = pd.read_pickle(depends_on)
    # calculate rolling mean
    z = df.rolling(window=30, center=True).mean()
    with open(produces, 'wb') as file:
        pickle.dump(z, file)



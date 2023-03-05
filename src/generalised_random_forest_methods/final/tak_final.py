import pytask
from generalised_random_forest_methods.config import BLD
from generalised_random_forest_methods.config import GROUPS
from generalised_random_forest_methods.config import SRC

import pandas as pd
from generalised_random_forest_methods.final import plot_regression_by_age
from generalised_random_forest_methods.utilities import read_yaml
from generalised_random_forest_methods.analysis.model import load_model


for group in GROUPS:

    kwargs = {
        "group": group,
        "depends_on": {"predictions": BLD / "python" / "predictions" / f"{group}.csv"},
        "produces": BLD / "python" / "figures" / f"smoking_by_{group}.png",
    }

    @pytask.mark.depends_on(
        {
            "data_info": SRC / "data_management" / "data_info.yaml",
            "data": BLD / "python" / "data" / "data_clean.csv",
        }
    )
    @pytask.mark.task(id=group, kwargs=kwargs)
    def task_plot_regression_python(depends_on, group, produces):
        data_info = read_yaml(depends_on["data_info"])
        data = pd.read_csv(depends_on["data"])
        predictions = pd.read_csv(depends_on["predictions"])
        fig = plot_regression_by_age(data, data_info, predictions, group)
        fig.write_image(produces)


@pytask.mark.depends_on(BLD / "python" / "models" / "model.pickle")
@pytask.mark.produces(BLD / "python" / "tables" / "estimation_results.tex")
def task_create_estimation_results_python(depends_on, produces):
    model = load_model(depends_on)
    table = model.summary().as_latex()
    with open(produces, "w") as f:
        f.writelines(table)


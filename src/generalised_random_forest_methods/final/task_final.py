import pytask
from generalised_random_forest_methods.config import BLD
from generalised_random_forest_methods.config import GROUPS
from generalised_random_forest_methods.config import SRC

import pandas as pd
from generalised_random_forest_methods.final.plot import treatment_effect_plot


@pytask.mark.depends_on(BLD / "python" / "data" / "z_rolling_means.pkl")
@pytask.mark.produces(BLD / "python" / "plots" / "treatment_effects_plot.png")
def task_plot_treatment_effects(depends_on, produces):
    z = pd.read_pickle(depends_on)
    plot = treatment_effect_plot(z)
    plot.savefig(produces)


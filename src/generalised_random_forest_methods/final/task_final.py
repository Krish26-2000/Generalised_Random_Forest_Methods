import pandas as pd
import pytask
from generalised_random_forest_methods.config import BLD
from generalised_random_forest_methods.config import GROUPS
from generalised_random_forest_methods.config import SRC
from generalised_random_forest_methods.final.plot import treatment_effect_plot
from generalised_random_forest_methods.final.plot import treatment_effect_plot2


@pytask.mark.depends_on(BLD / "python" / "data" / "z_rolling_means.pkl")
@pytask.mark.produces(BLD / "python" / "plots" / "treatment_effects_plot.png")
def task_plot_treatment_effects(depends_on, produces):
    z = pd.read_pickle(depends_on)
    plot = treatment_effect_plot(z)
    plot.savefig(produces)


@pytask.mark.depends_on(BLD / "python" / "data" / "z_rolling_means2.pkl")
@pytask.mark.produces(BLD / "python" / "plots" / "children_treatment_effects_plot.png")
def task_plot_treatment_effects2(depends_on, produces):
    z2 = pd.read_pickle(depends_on)
    plot = treatment_effect_plot2(z2)
    plot.savefig(produces)

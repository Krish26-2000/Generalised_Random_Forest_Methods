import pandas as pd
import pytask
from generalised_random_forest_methods.utilities import read_yaml
from generalised_random_forest_methods.config import BLD, SRC
from generalised_random_forest_methods.data_management.clean_data import (
    ipums_data, 
)
from generalised_random_forest_methods.data_management.clean_data import (
    clean_data,
)
from generalised_random_forest_methods.data_management.clean_data import (
    create_variables,
)

url = (
    "https://www.dropbox.com/s/luil3jn6m4jjdfj/usa_00006.csv?dl=1"
)

#@pytask.mark.depends_on(url)
@pytask.mark.produces(BLD / "python" / "usa_00006.csv")
def task_save_ipums_data(produces):
    """Import IPUMS data from Dropbox folder and save in the BLD folder

    Args: 
        produces: Specify the folder path to locate the file

    Returns:
        The IPUMS dataset
    
    """
    data = ipums_data(url)
    data.to_csv(produces)

@pytask.mark.depends_on(
    {
        "ipums_data": BLD / "python" / "usa_00006.csv",
        "data_info": SRC / "data_management" / "data_info.yaml",
    }
)
@pytask.mark.produces(BLD / "python" / "data" / "clean_df.pkl")
def task_clean_df(depends_on, produces):
    """

    Args:
        depends_on:
        produces:

    Returns:

    """
    data_info = read_yaml(depends_on["data_info"])
    data = pd.read_csv(depends_on["ipums_data"])
    clean_df = clean_data(data, data_info)
    clean_df.to_pickle(produces)


@pytask.mark.depends_on(BLD / "python" / "data" / "clean_df.pkl")
@pytask.mark.produces(BLD / "python" / "data" / "final_df.pkl")
def task_create_variables(depends_on, produces):
    """

    Args:
        depends_on:
        produces:

    Returns:

    """
    clean_df = pd.read_pickle(depends_on)
    final_df = create_variables(clean_df)
    final_df.to_pickle(produces)

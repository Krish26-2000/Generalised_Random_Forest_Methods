import pandas as pd
import pytask
from generalised_random_forest_methods.utilities import read_yaml
from generalised_random_forest_methods.config import BLD, SRC
from generalised_random_forest_methods.data_management.clean_data import (
    ipums_data, clean_data, create_columns,create_var_df
)
from generalised_random_forest_methods.config import TASK_1
from generalised_random_forest_methods.config import TASK_2

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
    final_df = create_columns(clean_df)
    final_df.to_pickle(produces)


for index, group in enumerate(TASK_1):

    kwargs = {
        "index":index,
        "produces": BLD / "python" / "data" / f"task_1_{group}.pkl",
    }

    @pytask.mark.depends_on(BLD / "python" / "data" / "final_df.pkl")
    @pytask.mark.task(id=index, kwargs=kwargs)
    def task_create_var_df(depends_on, produces, index):
        """

        Args:
            depends_on:
            produces:
            index:

        Returns:

        """
        data = pd.read_pickle(depends_on)
        variable_data = create_var_df()
        data[variable_data[index]].to_pickle(produces)

import pandas as pd
import pytask
from generalised_random_forest_methods.config import BLD, SRC
from generalised_random_forest_methods.data_management.clean_data import (
    ipums_data, 
)
from generalised_random_forest_methods.data_management.clean_data import (
    clean_data,
)

url = (
    "https://www.dropbox.com/s/0gsml2tasys0deu/usa_00006.csv?dl=1"
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

@pytask.mark.depends_on(BLD/ "python" / "usa_00006.csv")
@pytask.mark.produces(BLD / "python" / "data"/ "final_df.pkl")
def task_final_df(depends_on, produces):
    """

    Args:
        depends_on:
        produces:

    Returns:

    """
    data = pd.read_csv(depends_on)
    data = clean_data(data)
    data.to_pickle(produces)
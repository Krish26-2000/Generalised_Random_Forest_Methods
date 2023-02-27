import pytask
from generalised_random_forest_methods.config import BLD, SRC
from generalised_random_forest_methods.data_management.clean_data import (
    ipums_data, 
)

url = (
    "https://www.dropbox.com/s/0gsml2tasys0deu/usa_00006.csv?dl=1"
)

#@pytask.mark.depends_on(url)
@pytask.mark.produces(BLD / "python" / "usa_00004.csv")
def task_save_ipums_data(produces):
    """Import IPUMS data from Dropbox folder and save in the BLD folder

    Args: 
        produces: Specify the folder path to locate the file

    Returns:
        The IPUMS dataset
    
    """
    data = ipums_data(url)
    data.to_csv(produces)

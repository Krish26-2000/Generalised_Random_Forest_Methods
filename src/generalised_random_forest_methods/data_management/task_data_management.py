import pytask
from generalised_random_forest_methods.config import BLD, SRC
from generalised_random_forest_methods.data_management.clean_data import (
    ipums_2019_data, ipums_2015_data,
)

url_1 = (
    "https://www.dropbox.com/s/0gsml2tasys0deu/usa_00004.csv?dl=1"
)

#@pytask.mark.depends_on(url)
@pytask.mark.produces(BLD / "python" / "usa_00004.csv")
def task_save_ipums__2019_data(produces):
    """Import IPUMS data from Dropbox folder and save in the BLD folder

    Args: 
        produces: Specify the folder path to locate the file

    Returns:
        The IPUMS dataset
    
    """
    data_2019 = ipums_2019_data(url_1)
    data_2019.to_csv(produces)

url_2 = (
    "https://www.dropbox.com/s/0gsml2tasys0deu/usa_00005.csv?dl=1"
)

#@pytask.mark.depends_on(url)
@pytask.mark.produces(BLD / "python" / "usa_00005.csv")
def task_save_ipums__2015_data(produces):
    """Import IPUMS data from Dropbox folder and save in the BLD folder

    Args: 
        produces: Specify the folder path to locate the file

    Returns:
        The IPUMS dataset
    
    """
    data_2015 = ipums_2015_data(url_2)
    data_2015.to_csv(produces)


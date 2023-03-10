import pandas as pd
import pytask
from generalised_random_forest_methods.utilities import read_yaml
from generalised_random_forest_methods.config import BLD, SRC, NO_LONG_TASKS
from generalised_random_forest_methods.data_management.clean_data import (
    ipums_data, clean_data, create_columns, return_var_dict, causal_model,train_test_data,
)
from generalised_random_forest_methods.config import TASK_1

url = (
    "https://www.dropbox.com/s/luil3jn6m4jjdfj/usa_00006.csv?dl=1"
)


#@pytask.mark.depends_on(url)
# @pytask.mark.skipif(NO_LONG_TASKS, reason="skipping long task")
@pytask.mark.produces(BLD / "python" / "usa_00006.csv")
def tsk_save_ipums_data(produces):
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


@pytask.mark.depends_on(BLD / "python" / "data" / "final_df.pkl")
@pytask.mark.produces(BLD / "python" / "plots" / "causal_model.png")
def task_casual_model(depends_on, produces):
    """

    Args:
        depends_on:
        produces:

    Returns:

    """
    data = pd.read_pickle(depends_on)
    variable_dict = return_var_dict()
    model = causal_model(data, **variable_dict)
    model.view_model(file_name=str(produces)[:-4])


@pytask.mark.depends_on(BLD / "python" / "data" / "final_df.pkl")
@pytask.mark.produces(
    [BLD / "python" / "data" / "train_df.pkl",
     BLD / "python" / "data" / "test_df.pkl"]
                      )
def task_train_test_data(depends_on, produces):
    """

    Args:
        depends_on:
        produces:

    Returns:

    """
    data = pd.read_pickle(depends_on)
    train_test_df = train_test_data(data)
    for i, df in enumerate(train_test_df):
        df.to_pickle(produces[i])


for index, group in enumerate(TASK_1):

    kwargs = {
        "group":group,
        "produces": BLD / "python" / "data" / f"{group}.pkl",
    }

    df = group.split('_')[1]

    @pytask.mark.depends_on(BLD / "python" / "data" / f"{df}_df.pkl")
    @pytask.mark.task(id=index, kwargs=kwargs)
    def task_train_test_vars(depends_on, produces, group):
        """

        Args:
            depends_on:
            produces:
            group:

        Returns:

        """
        data = pd.read_pickle(depends_on)
        variable_dict = return_var_dict()
        variable = group.split('_')[0]
        data[variable_dict[variable]].to_pickle(produces)



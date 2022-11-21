import numpy as np
import pandas as pd
import time
import copy
from carla.recourse_methods import CCHVAE

def cchvae_function(data, cchvae_model, x_carla_df):
    """
    DESCRIPTION:    Method that runs the CCHVAE model
    
    INPUT
    data:           Dataset object containing the dataset and related relevant information
    cchvae_model:   CARLA CCHVAE model
    x_carla_df:     Instance of interest loaded with the preprocessing for the CARLA framework

    OUTPUT
    cf:             Counterfactual instance obtained from the CCHVAE method
    run_time:       Run time of the CCHVAE method        
    """
    data_with_target = copy.deepcopy(data.train_df)
    # data_with_target[data.label_str[0]] = data.train_target
    start_time = time.time()
    cf_df = cchvae_model.get_counterfactuals(x_carla_df)
    if isinstance(cf_df, pd.Series) or isinstance(cf_df, pd.DataFrame):
        if cf_df.isnull().values.any():
            cf = None
            print(f'CCHVAE: Could not find feasible CF!')
        else:
            cf_df = data.from_carla(cf_df)
            cf = np.array(cf_df)[0]
    elif cf_df is None:
        cf = None
        print(f'CCHVAE: Could not find feasible CF!')
    elif cf_df is not None:
        if np.isnan(np.sum(cf_df)):
            cf = None
            cf_df = None
            print(f'CCHVAE: Could not find feasible CF!')
    end_time = time.time() 
    run_time = end_time - start_time
    return cf, run_time
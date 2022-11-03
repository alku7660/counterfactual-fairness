import numpy as np
import pandas as pd
import time
import copy
from carla.recourse_methods import CCHVAE

def cchvae_function(data, carla_model, x_carla_pd):
    """
    DESCRIPTION:    Method that runs the CCHVAE model
    
    INPUT
    data:           Dataset object containing the dataset and related relevant information
    carla_model:    CARLA framework model
    x_carla_pd:     Instance of interest loaded with the preprocessing for the CARLA framework

    OUTPUT
    cf:             Counterfactual instance obtained from the CCHVAE method
    run_time:       Run time of the CCHVAE method        
    """
    data_with_target = copy.deepcopy(data.train_pd)
    data_with_target[data.label_str[0]] = data.train_target
    start_time = time.time()
    dict_cchvae = {'data_name': data.name,'p_norm':2,'vae_params':{'layers':[len(carla_model.feature_input_order),int(len(carla_model.feature_input_order)/2)]}}
    cchvae_model = CCHVAE(carla_model, dict_cchvae, data_with_target)
    cf_pd = cchvae_model.get_counterfactuals(x_carla_pd)
    if isinstance(cf_pd, pd.Series) or isinstance(cf_pd, pd.DataFrame):
        if cf_pd.isnull().values.any():
            cf = None
            print(f'CCHVAE: Could not find feasible CF!')
        else:
            cf_pd = data.from_carla_to_jce(cf_pd)
            cf = np.array(cf_pd)[0]
    elif cf_pd is None:
        cf = None
        print(f'CCHVAE: Could not find feasible CF!')
    elif cf_pd is not None:
        if np.isnan(np.sum(cf_pd)):
            cf = None
            cf_pd = None
            print(f'CCHVAE: Could not find feasible CF!')
    end_time = time.time() 
    run_time = end_time - start_time
    return cf, run_time
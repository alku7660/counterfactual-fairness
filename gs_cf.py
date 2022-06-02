import numpy as np
import pandas as pd
import time
import copy
from carla.recourse_methods import GrowingSpheres

def gs_function(data, carla_model, x_carla_pd):
    """
    Method that runs the GS model
    """
    start_time = time.time()
    gs_model = GrowingSpheres(carla_model)
    cf_pd = gs_model.get_counterfactuals(x_carla_pd)
    if isinstance(cf_pd, pd.Series) or isinstance(cf_pd, pd.DataFrame):
        if cf_pd.isnull().values.any():
            cf = None
            print(f'GS: Could not find feasible CF!')
        else:
            cf_pd = data.from_carla_to_jce(cf_pd)
            cf = np.array(cf_pd)[0]
    elif np.isnan(np.sum(cf_pd)):
        cf = None
        print(f'GS: Could not find feasible CF!')
    end_time = time.time() 
    run_time = end_time - start_time
    return cf, run_time

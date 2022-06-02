import numpy as np
import pandas as pd
import time
import copy
from carla.recourse_methods import Dice

def dice_function(data, carla_model, x_carla_pd):
    """
    Method that runs the DiCE model
    """
    start_time = time.time()
    dice_model = Dice(carla_model, {'desired_class': int(1-data.undesired_class)})
    cf_pd = dice_model.get_counterfactuals(x_carla_pd)
    if isinstance(cf_pd, pd.Series) or isinstance(cf_pd, pd.DataFrame):
        if cf_pd.isnull().values.any():
            cf = None
            print(f'DICE: Could not find feasible CF!')
        else:
            cf_pd = data.from_carla_to_jce(cf_pd)
            cf = np.array(cf_pd)[0]
    elif cf_pd is None:
        cf = None
        print(f'DICE: Could not find feasible CF!')
    elif cf_pd is not None:
        if np.isnan(np.sum(cf_pd)):
            cf = None
            cf_pd = None
            print(f'DICE: Could not find feasible CF!')
    end_time = time.time() 
    run_time = end_time - start_time
    return cf, run_time
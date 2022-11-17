"""
Nearest Neighbor (NN)
"""

"""
Imports
"""

import numpy as np
import time
from support import verify_feasibility

def near_neigh(x, x_label, data, mutability_check=True):
    """
    DESCRIPTION:        Returns the nearest counterfactual with respect to instance of interest x
    
    INPUT:
    x:                  Instance of interest
    x_label:            Label of instance of interest x
    data:               Dataset object
    mutability_check:   Whether to check or not the mutable features

    OUTPUT:
    nt_cf:              Nearest neighbor counterfactual to the instance of interest x
    nn_time:            Nearest neighbor counterfactual run time
    """
    start_time = time.time()
    nt_cf = None
    for i in data.train_sorted:
        if i[2] != x_label and verify_feasibility(x, i[0], data.feat_mutable, data.feat_type, data.feat_step, data.feat_dir, mutability_check) and not np.array_equal(x, i[0]):
            nt_cf = i[0]
            break
    if nt_cf is None:
        print(f'NT could not find a feasible CF!: There is no feasible NN CF available (Looking for closest counterfactual labeled training observation)')
        for i in data.train_sorted:
            if i[2] != x_label and not np.array_equal(x, i[0]):
                nt_cf = i[0]
        end_time = time.time()
        return nt_cf, end_time - start_time
    end_time = time.time()
    nt_time = end_time - start_time + data.training_sort_time
    return nt_cf, nt_time

def nn_model(prev_nn, x, x_label, data, model, mutability_check=True):
    """
    DESCRIPTION:        Returns the nearest counterfactual with respect to instance of interest x
    
    INPUT:
    prev_nn:            NT instance using training dataset information and label
    x:                  Instance of interest
    x_label:            Label of instance of interest x
    data:               Dataset model
    model:              Trained model to verify different predicted label from the test dataset
    mutability_check:   Whether to check or not the mutable features

    OUTPUT:
    nt_cf:              Nearest neighbor counterfactual to the instance of interest x
    """
    nt_cf = prev_nn
    for i in data.train_sorted:
        if i[2] != x_label and model.predict(i[0].reshape(1,-1)) != x_label and verify_feasibility(x,i[0],data.feat_mutable,data.feat_type,data.feat_step,data.feat_dir,mutability_check) and not np.array_equal(x,i[0]):
                nt_cf = i[0]
                break
    return nt_cf
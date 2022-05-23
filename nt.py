"""
Nearest Neighbor (NN)
"""

"""
Imports
"""

import numpy as np
import time
from support import verify_feasibility

def nn(x,x_label,data,mutability_check=True):
    """
    Function that returns the nearest counterfactual with respect to instance of interest x
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input data: Dataset object
    Input mutability_check: Whether to check or not the mutable features
    Output nt_cf: Minimum observable counterfactual to the instance of interest x
    """
    start_time = time.time()
    nt_cf = None
    for i in data.train_sorted:
        if i[2] != x_label and verify_feasibility(x,i[0],data.feat_mutable,data.feat_type,data.feat_step,data.feat_dir,mutability_check) and not np.array_equal(x,i[0]):
            nt_cf = i[0]
            break
    if nt_cf is None:
        print(f'NT could not find a feasible CF!: There is no feasible NN CF available (None output)')
        end_time = time.time()
        return nt_cf, end_time - start_time
    end_time = time.time()
    nn_time = end_time - start_time
    return nt_cf, nn_time

def nn_model(prev_nn,x,x_label,data,model,mutability_check=True):
    """
    Function that returns the nearest counterfactual with respect to instance of interest x
    Input prev_nn: NT instance using training dataset information and label
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input data: Dataset model
    Input model: Trained model to verify different predicted label from the test dataset
    Input mutability_check: Whether to check or not the mutable features
    Output nt_cf: Minimum observable counterfactual to the instance of interest x
    """
    nt_cf = prev_nn
    for i in data.train_sorted:
        if i[2] != x_label and model.predict(i[0].reshape(1,-1)) != x_label and verify_feasibility(x,i[0],data.feat_mutable,data.feat_type,data.feat_step,data.feat_dir,mutability_check) and not np.array_equal(x,i[0]):
                nt_cf = i[0]
                break
    return nt_cf
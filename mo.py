"""
Minimum Observable (MO)
"""

"""
Imports
"""
import numpy as np
import time
from address_distance_feasibility import verify_feasibility
from address_distance_feasibility import sort_data_distance

#Minimum Observable method
def min_obs(x,x_label,data):
    """
    Function that returns the minimum observable counterfactual with respect to instance of interest x
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input data: Dataset object
    Output mo_cf: Minimum observable counterfactual to the instance of interest x
    Output mo_cf_dist_x: Distance between mo_cf and instance of interest x
    """
    start_time = time.time()
    mo_cf = None
    all_data = np.vstack((data.jce_train_np,data.jce_test_undesired_np))
    all_labels = np.hstack((data.train_target,data.test_undesired_target))
    data_distance_mo = sort_data_distance(x,all_data,all_labels)
    for i in data_distance_mo:
        if i[2] != x_label and verify_feasibility(x,i[0],data.feat_mutable,data.feat_type,data.feat_step) and not np.array_equal(x,i[0]):
            mo_cf = i[0]
            break
    if mo_cf is None:
        print(f'MO could not find a feasible CF!')
        end_time = time.time()
        return mo_cf, end_time - start_time
    end_time = time.time()
    mo_time = end_time - start_time
    return mo_cf, mo_time
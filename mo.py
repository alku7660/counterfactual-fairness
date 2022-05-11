"""
Minimum Observable (MO)
"""

"""
Imports
"""
import numpy as np
import time
from support import sort_data_distance

def verify_feasibility(x,cf,mutable_feat,feat_type,feat_step,feat_dir,mutability_check):
    """
    Method that indicates whether cf is a feasible counterfactual with respect to x and the feature mutability
    Input x: Instance of interest
    Input cf: Counterfactual to be evaluated
    Input mutable_feat: Vector indicating mutability of the features of x
    Input feat_type: Type of the features used
    Input feat_step: Feature plausible change step size
    Input feat_dir: Directionality of the features
    Input mutability_check: Whether to check or not the mutable features    
    Output: Boolean value indicating whether cf is a feasible counterfactual with regards to x and the feature mutability vector
    """
    toler = 0.000001
    feasibility = True
    for i in range(len(feat_type)):
        if feat_type[i] == 'bin':
            if not np.isclose(cf[i], [0,1],atol=toler).any():
                feasibility = False
                break
        elif feat_type[i] == 'num-ord':
            possible_val = np.linspace(0,1,int(1/feat_step[i]+1),endpoint=True)
            if not np.isclose(cf[i],possible_val,atol=toler).any():
                feasibility = False
                break  
        else:
            if cf[i] < 0-toler or cf[i] > 1+toler:
                feasibility = False
                break
        vector = cf - x
        if feat_dir[i] == 0 and vector[i] != 0:
            feasibility = False
            break
        elif feat_dir[i] == 'pos' and vector[i] < 0:
            feasibility = False
            break
        elif feat_dir[i] == 'neg' and vector[i] > 0:
            feasibility = False
            break
    if mutability_check:
        if not np.array_equal(x[np.where(mutable_feat == 0)],cf[np.where(mutable_feat == 0)]):
            feasibility = False
    return feasibility

#Minimum Observable method
def min_obs(x,x_label,data,mutability_check=True):
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
        if i[2] != x_label and verify_feasibility(x,i[0],data.feat_mutable,data.feat_type,data.feat_step,data.feat_dir,mutability_check) and not np.array_equal(x,i[0]):
            mo_cf = i[0]
            break
    if mo_cf is None:
        print(f'MO could not find a feasible CF!')
        end_time = time.time()
        return mo_cf, end_time - start_time
    end_time = time.time()
    mo_time = end_time - start_time
    return mo_cf, mo_time
"""
Support functions & imports
"""

import os
import numpy as np
path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/Datasets/'
results_cf_obj_dir = str(path_here)+'/Results/cf_obj/'
results_cf_plots_dir = str(path_here)+'/Results/cf_plots/'
results_mace_dir = str(path_here)+'/Results/mace/'

def euclidean(x1,x2):
    """
    Calculation of the euclidean distance between two different instances
    Input x1: Instance 1
    Input x2: Instance 2
    Output euclidean distance between x1 and x2
    """
    return np.sqrt(np.sum((x1-x2)**2))

def sort_data_distance(x,data,data_label):
    """
    Function to organize dataset with respect to distance to instance x
    Input x: Instance (can be the instane of interest or a synthetic instance)
    Input data: Training dataset
    Input data_label: Training dataset label
    Output data_sorted_distance: Training dataset sorted by distance to the instance of interest x
    """
    sort_data_distance = []
    for i in range(len(data)):
        dist = euclidean(data[i],x)
        sort_data_distance.append((data[i],dist,data_label[i]))      
    sort_data_distance.sort(key=lambda x: x[1])
    return sort_data_distance

def verify_feasibility(x,cf,mutable_feat,feat_type,feat_step,feat_dir):
    """
    Method that indicates whether cf is a feasible counterfactual with respect to x and the feature mutability
    Input x: Instance of interest
    Input cf: Counterfactual to be evaluated
    Input mutable_feat: Vector indicating mutability of the features of x
    Input feat_type: Type of the features used
    Input feat_step: Feature plausible change step size
    Input feat_dir: Directionality of the features    
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
    if not np.array_equal(x[np.where(mutable_feat == 0)],cf[np.where(mutable_feat == 0)]):
        feasibility = False
    return feasibility

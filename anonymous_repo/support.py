"""
Imports
"""
import os
import pickle
import numpy as np

path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/Datasets/'
results_cf_obj_dir = str(path_here)+'/Results/cf_obj/'
results_cf_obj_method_dir = str(path_here)+'/Results/cf_obj_method/'
results_cf_plots_dir = str(path_here)+'/Results/cf_plots/'
results_grid_search = str(path_here)+'/Results/grid_search/'

def euclidean(x1, x2):
    """
    Calculation of the euclidean distance between two different instances
    Input x1: Instance 1
    Input x2: Instance 2
    Output euclidean distance between x1 and x2
    """
    return np.sqrt(np.sum((x1-x2)**2))

def sort_data_distance(x, data, data_label):
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
        sort_data_distance.append((data[i], dist, data_label[i]))      
    sort_data_distance.sort(key=lambda x: x[1])
    return sort_data_distance

def verify_feasibility(x, cf, data, mutability_check=True):
    """
    Method that indicates whether cf is a feasible counterfactual with respect to x and the feature mutability
    Input x: Instance of interest
    Input cf: Counterfactual to be evaluated
    Input data: Data object   
    Output: Boolean value indicating whether cf is a feasible counterfactual with regards to x and the feature mutability vector
    """
    """
    Method that indicates whether the cf is a feasible counterfactual with respect to x, feature mutability and directionality
    """
    toler = 0.000001
    feasibility = True
    for i in range(len(data.feat_type)):
        if data.feat_type[i] == 'bin' or data.feat_type[i] == 'cat':
            if not np.isclose(cf[i], [0,1], atol=toler).any():
                feasibility = False
                break
        elif data.feat_type[i] == 'ord':
            possible_val = np.linspace(0, 1, int(1/data.feat_step[i]+1), endpoint=True)
            if not np.isclose(cf[i], possible_val, atol=toler).any():
                feasibility = False
                break  
        else:
            if cf[i] < 0-toler or cf[i] > 1+toler:
                feasibility = False
                break
        if mutability_check:
            vector = cf - x
            if data.feat_dir[i] == 0 and vector[i] != 0:
                feasibility = False
                break
            elif data.feat_dir[i] == 'pos' and vector[i] < 0:
                feasibility = False
                break
            elif data.feat_dir[i] == 'neg' and vector[i] > 0:
                feasibility = False
                break
    if mutability_check:
        if not np.array_equal(x[np.where(data.feat_mutable == 0)], cf[np.where(data.feat_mutable == 0)]):
            feasibility = False
    return feasibility

def save_obj(evaluator_obj, file_name):
    """
    Method to store an Evaluator object containing the evaluation results for all the instances of a given dataset
    """
    with open(results_cf_obj_method_dir+file_name, 'wb') as output:
        pickle.dump(evaluator_obj, output, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name):
    """
    Method to read an Evaluator object containing the evaluation results for all the instances of a given dataset
    """
    with open(results_cf_obj_method_dir+file_name, 'rb') as input:
        evaluator_obj = pickle.load(input)
    return evaluator_obj
"""
Imports
"""
import warnings
warnings.filterwarnings("ignore")
from juice import JUICE
from mixed_features_justification import verify_justification
from data_load_juice import load_model_dataset
import pickle
import numpy as np
import pandas as pd
import os
path_here = os.path.abspath('')
dataset_dir = str(path_here)+'/Datasets/'
results_cf_obj_dir = str(path_here)+'/Results/cf_obj/'

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
        dist = np.linalg.norm(data[i] - x)
        sort_data_distance.append((data[i],dist,data_label[i]))      
    sort_data_distance.sort(key=lambda x: x[1])
    return sort_data_distance

def proximity(normal_x,normal_cf):
    """
    Method that calculates the distance between the instance of interest and the counterfactual given
    Input normal_x: The normalized instance of interest
    Input normal_cf: The normalized counterfactual instance
    """
    cf_proximity = np.round_(np.linalg.norm(normal_x - normal_cf),3)
    return cf_proximity

def sparsity(data,x,cf):
    """
    Function that calculates sparsity for a given counterfactual according to x
    Sparsity: 1 - the fraction of features changed in the cf. Takes the value of 1 if the number of changed features is 1.
    Input data: The data object with the feature information regarding plausibility, mutability, directionality
    Input x: The (could be normalized) instance of interest
    Input cf: The (could be normalized) counterfactual instance        
    """
    unchanged_features = np.sum(np.equal(x,cf))
    categories_feat_changed = data.feat_cat[np.where(np.equal(x,cf) == False)[0]]
    len_categories_feat_changed_unique = len([i for i in np.unique(categories_feat_changed) if 'cat' in i])
    unchanged_features += len_categories_feat_changed_unique
    n_changed = len(x) - unchanged_features
    if n_changed == 1:
        cf_sparsity = 1.000
    else:
        cf_sparsity = np.round_(1 - n_changed/len(x),3)
    return cf_sparsity

def feasibility(data,normal_x,normal_cf):
    """
    Method that indicates whether cf is a feasible counterfactual with respect to x
    Input data: The data object with the feature information regarding plausibility, mutability, directionality
    Input normal_x: The normalized instance of interest
    Input normal_cf: The normalized counterfactual instance
    """
    toler = 0.000001
    cf_feasibility = True
    for i in range(len(data.feat_type)):
        if data.feat_type[i] == 'bin':
            if not np.isclose(normal_cf[i],[0,1],atol=toler).any():
                cf_feasibility = False
                break
        elif data.feat_type[i] == 'num-ord':
            possible_val = np.linspace(0,1,int(1/data.feat_step[i]+1),endpoint=True)
            if not np.isclose(normal_cf[i],possible_val,atol=toler).any():
                cf_feasibility = False
                break
        else:
            if normal_cf[i] < 0-toler or normal_cf[i] > 1+toler:
                cf_feasibility = False
                break
        vector = normal_cf - normal_x
        if data.feat_dir[i] == 0 and vector[i] != 0:
            cf_feasibility = False
            break
        elif data.feat_dir[i] == 'pos' and vector[i] < 0:
            cf_feasibility = False
            break
        elif data.feat_dir[i] == 'neg' and vector[i] > 0:
            cf_feasibility = False
            break
    if not np.array_equal(normal_x[np.where(data.feat_mutable == 0)],normal_cf[np.where(data.feat_mutable == 0)]):
        cf_feasibility = False
    return cf_feasibility
    
def justification(normal_cf,cf_label,nn_to_cf,n_feat,model,data):
    """
    Method that finds whether the instance of interest is justified or not
    Input normal_cf: The normalized counterfactual instance
    Input cf_label: Label of the counterfactual
    Input nn_to_cf: Closest nearest neighbor to the CF
    Input n_feat: Number of features to generate per feature in the continuous feature space
    Input model: Model object
    Input data: Data object
    """
    justifier_instance, cf_justification = verify_justification(normal_cf,cf_label,nn_to_cf,n_feat,model,data)
    justifier_instance_pd = pd.DataFrame(data=[justifier_instance],columns=data.juice_all_cols)
    return justifier_instance_pd, cf_justification

def nn_to_cf_search(normal_cf,data,x_label,idx):
    """
    Method that finds the nearest neighbor to the CF
    Input normal_cf: The normalized counterfactual instance
    Input data: The data object with juice columns, and training dataset
    Input x_label: Label of the instance of interest
    """
    if normal_cf is not None and not np.isnan(np.sum(normal_cf)):
        if isinstance(normal_cf,pd.DataFrame):
            normal_cf = normal_cf
        elif isinstance(normal_cf,pd.Series):
            cf_np = normal_cf.to_numpy()
            normal_cf = pd.DataFrame(data = [cf_np], index = [idx], columns = data.juice_all_cols) 
        else:
            normal_cf = pd.DataFrame(data = [normal_cf], index = [idx], columns = data.juice_all_cols)
        sorted_train_normal_cf = sort_data_distance(normal_cf,data.juice_train_np,data.train_target)
        for i in sorted_train_normal_cf:
            if i[2] != x_label:
                nn_to_normal_cf = i[0]
                label_nn_to_normal_cf = i[2]
                break
    return nn_to_normal_cf, label_nn_to_normal_cf

def save_obj(evaluator_obj,file_name):
    """
    Method to store an Evaluator object containing the evaluation results for all the instances of a given dataset
    """
    with open(results_cf_obj_dir+file_name, 'wb') as output:
        pickle.dump(evaluator_obj, output, pickle.HIGHEST_PROTOCOL)

datasets = ['synthetic_disease','synthetic_athlete','ionosphere','compass','credit','adult','german','heart']       # List of the datasets to be analyzed ['synthetic_disease','synthetic_athlete','ionosphere','compass','credit','adult','german','heart']
step = 0.01                # Step size to change continuous features
train_fraction = 0.7       # Percentage of examples to use for training
n_feat = 50                # Number of examples to generate synthetically per feature
seed_int = 54321           # Seed integer value
perc = 1                   # Percentage of test samples to consider for the counterfactuals search
np.random.seed(seed_int)

for data_str in datasets:

    data, model = load_model_dataset(data_str,train_fraction,seed_int,step,path_here)

    print(f'---------------------------------------')  
    print(f'                    Dataset: {data_str}')
    print(f'        Train dataset shape: {data.train_pd.shape}')
    print(f'         Test dataset shape: {data.test_undesired_pd.shape}')
    print(f' JUICE model train accuracy: {np.round_(model.juice_sel.score(data.juice_train_pd,data.train_target),2)}')
    print(f'  JUICE model test accuracy: {np.round_(model.juice_sel.score(data.juice_test_undesired_pd,data.test_undesired_target),2)}')
    print(f'---------------------------------------')

    # save_obj(model,data_str+'_model.pkl')
    # save_obj(data,data_str+'_data.pkl')
    test_undesired_index = data.test_undesired_pd.index.to_list()

    for i in range(int(len(test_undesired_index)*perc)):

        print(f'--------------------------------------------------------------')
        print(f'                    Dataset: {data_str}')
        print(f'                    Test instance number: {i}')
        print(f'--------------------------------------------------------------')

        idx = test_undesired_index[i]
        x_juice_pd = data.juice_test_undesired_pd.loc[idx]
        x_juice_np = x_juice_pd.to_numpy()
        x_label = model.juice_sel.predict(x_juice_np.reshape(1,-1))
        data.add_sorted_train_data(x_juice_pd)

        juice_prox_cf, instance_juice_prox, just_juice_prox, found_justifiable_juice_prox, juice_prox_time = JUICE(x_juice_np,x_label,data,model.juice_sel,'proximity') # Refer to juice.py for details
        print(f'  JUICEP (time (s): {np.round_(juice_prox_time,2)})')
        print(f'---------------------------')

        juice_spar_cf, instance_juice_spar, just_juice_spar, found_justifiable_juice_spar, juice_spar_time = JUICE(x_juice_np,x_label,data,model.juice_sel,'sparsity') # Refer to juice.py for details
        sparsity_changed_feat, juice_spar_cf = juice_spar_cf[1], juice_spar_cf[0]
        print(f'  JUICES (time (s): {np.round_(juice_spar_time,2)})')
        print(f'---------------------------')

        if juice_prox_cf is not None:

            nn_to_juicep_cf, nn_to_juicep_cf_label = nn_to_cf_search(juice_prox_cf,data,x_label,idx)
            nn_to_juices_cf, nn_to_juices_cf_label = nn_to_cf_search(juice_spar_cf,data,x_label,idx)

            print(f'--------------------------------------------------------------')
            print(f'     DONE dataset: {data_str}, instance: {idx} Evaluation')      
            print(f'     JUICEP CF: {juice_prox_cf}')
            print(f'     JUICEP Proximity     : {proximity(x_juice_np,juice_prox_cf)}')
            print(f'     JUICEP Sparsity      : {sparsity(data,x_juice_np,juice_prox_cf)}')
            print(f'     JUICEP Feasibility   : {feasibility(data,x_juice_np,juice_prox_cf)}')         
            print(f'     JUICEP Justification : {just_juice_prox}')  
            print(f'     JUICEP Time          : {juice_prox_time}')   
            print(f'--------------------------------------------------------------')
            print(f'     JUICES CF: {juice_spar_cf}')
            print(f'     JUICES Proximity     : {proximity(x_juice_np,juice_spar_cf)}')
            print(f'     JUICES Sparsity      : {sparsity(data,x_juice_np,juice_spar_cf)}')
            print(f'     JUICES Feasibility   : {feasibility(data,x_juice_np,juice_spar_cf)}')         
            print(f'     JUICES Justification : {just_juice_spar}')  
            print(f'     JUICES Time          : {juice_spar_time}')   
            print(f'--------------------------------------------------------------')
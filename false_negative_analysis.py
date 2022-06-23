"""
Imports
"""
import warnings
warnings.filterwarnings("ignore")
from eval import Evaluator
from data_load import load_model_dataset
import pickle
import numpy as np
import pandas as pd
from carla_adapter import MyOwnDataSet, MyOwnModel
from support import path_here, results_cf_obj_dir

def save_obj(evaluator_obj,file_name):
    """
    Method to store an Evaluator object containing the evaluation results for all the instances of a given dataset
    """
    with open(results_cf_obj_dir+file_name, 'wb') as output:
        pickle.dump(evaluator_obj, output, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name):
    """
    Method to read an Evaluator object containing the evaluation results for all the instances of a given dataset
    """
    with open(results_cf_obj_dir+file_name, 'rb') as input:
        evaluator_obj = pickle.load(input)
    return evaluator_obj

datasets = ['adult','kdd_census','german','dutch','bank','compass','diabetes','student','oulad','law'] # Name of the dataset to be analyzed ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law']
models_to_run = ['nn','mo','rt','cchvae'] #['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice'] 'dutch','credit','diabetes','oulad','law'
step = 0.01                # Step size to change continuous features
train_fraction = 0.7       # Percentage of examples to use for training
n_feat = 50                # Number of examples to generate synthetically per feature
k = 50                     # Number of training dataset neighbors to consider for the correlation matrix calculation for the CIJN method
epsilon_ft = 0.01          # Epsilon corresponding to the rate of change in feature tweaking algorithm
seed_int = 54321           # Seed integer value
only_undesired_cf = 1      # Find counterfactuals only for negative (bad) class factuals
perc = 1                  # Percentage of false negative test samples to consider for the counterfactuals search

np.random.seed(seed_int)

for data_str in datasets:

    data, model = load_model_dataset(data_str,train_fraction,seed_int,step,path_here)
    carla_data = MyOwnDataSet(data)
    carla_model = MyOwnModel(carla_data,data,model)
    cf_evaluator = Evaluator(data,n_feat,models_to_run)

    print(f'---------------------------------------')  
    print(f'                    Dataset: {data_str}')
    print(f'        Train dataset shape: {data.train_pd.shape}')
    print(f'         Test dataset shape: {data.test_undesired_pd.shape}')
    print(f'   JCE model train accuracy: {np.round_(model.jce_sel.score(data.jce_train_pd,data.train_target),2)}')
    print(f'    JCE model test accuracy: {np.round_(model.jce_sel.score(data.jce_test_undesired_pd,data.test_undesired_target),2)}')
    print(f' CARLA model train accuracy: {np.round_(carla_model._mymodel.score(data.carla_train_pd,data.train_target),2)}')
    print(f'  CARLA model test accuracy: {np.round_(carla_model._mymodel.score(data.carla_test_pd,data.test_target),2)}')
    print(f'---------------------------------------')

    save_obj(model,data_str+'_fnr_model.pkl')
    save_obj(data,data_str+'_fnr_data.pkl')
    
    cf_evaluator.add_fairness_measures(data, model)
    desired_ground_truth_test_pd = data.jce_test_pd.loc[data.test_target != data.undesired_class]
    desired_ground_truth_target = data.test_target[data.test_target != data.undesired_class]
    predicted_label_desired_ground_truth_test_pd = model.jce_sel.predict(desired_ground_truth_test_pd)
    false_undesired_test_pd = desired_ground_truth_test_pd.loc[predicted_label_desired_ground_truth_test_pd == data.undesired_class]
    false_undesired_target = desired_ground_truth_target[predicted_label_desired_ground_truth_test_pd == data.undesired_class]
    for i in range(int(len(false_undesired_test_pd)*perc)):
        idx = false_undesired_test_pd.index.tolist()[i]
        x_instance = false_undesired_test_pd.loc[idx]
        x_target = false_undesired_target[i]
        print(f'---------------------------')
        print(f'     Dataset: {data_str}')
        print(f'  Test instance number: {i}')
        print(f'  Idx number: {idx}')
        print(f'  Total instances: {int(len(false_undesired_test_pd)*perc)}')
        print(f'---------------------------')
        x_jce_np = x_instance.to_numpy()
        x_carla_pd = data.test_pd.loc[idx].to_frame().T
        x_label = model.jce_sel.predict(x_jce_np.reshape(1,-1))
        data.add_sorted_train_data(x_instance)
        cf_evaluator.add_specific_x_data(idx,x_jce_np,x_carla_pd,x_label,x_target,data)
        cf_evaluator.evaluate_cf_models(x_jce_np,x_label,data,model,epsilon_ft,carla_model,x_carla_pd)
                    
    print(f'---------------------------')
    print(f'  DONE: {data_str} CF Evaluation')
    print(f'---------------------------')
    save_obj(cf_evaluator,data_str+'_fnr_eval.pkl')

print(f'---------------------------')
print(f'  DONE: All CFs and Datasets')
print(f'---------------------------')
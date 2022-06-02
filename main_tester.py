"""
Imports
"""
import warnings
warnings.filterwarnings("ignore")
import time
import copy
from nt import nn
from mo import min_obs
from rt import rf_tweak
from ft import feat_tweak
from juice import JUICE
from eval import Evaluator
from data_load import load_model_dataset
import pickle
import numpy as np
import pandas as pd
from carla import DataCatalog, MLModelCatalog
from carla.recourse_methods import Face, Dice, CCHVAE, GrowingSpheres
import tensorflow as tf
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

datasets = ['diabetes','student','oulad','law'] # Name of the dataset to be analyzed ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law']
models_to_run = ['nn','mutable-nn','mo','mutable-mo','rt','mutable-rt','cchvae','face'] #['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice']
step = 0.01                # Step size to change continuous features
train_fraction = 0.7       # Percentage of examples to use for training
n_feat = 50                # Number of examples to generate synthetically per feature
k = 50                     # Number of training dataset neighbors to consider for the correlation matrix calculation for the CIJN method
epsilon_ft = 0.01          # Epsilon corresponding to the rate of change in feature tweaking algorithm
seed_int = 54321           # Seed integer value
only_undesired_cf = 1      # Find counterfactuals only for negative (bad) class factuals
perc = 1             # Percentage of test samples to consider for the counterfactuals search
instances_per_feature_val = 20
instances_per_undesired_class = 100

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

    save_obj(model,data_str+'_model.pkl')
    save_obj(data,data_str+'_data.pkl')
    
    cf_evaluator.add_fairness_measures(data, model)
    evaluated_instances_pd = pd.DataFrame(columns=cf_evaluator.raw_data_cols)
    idx_instances_evaluated = []
    test_undesired_index = data.test_undesired_pd.index.to_list()
    for feat in data.feat_protected.keys():
        feat_unique_val_dict = {i:0 for i in data.test_pd[feat].unique()}
        for i in range(1,int(len(test_undesired_index)*perc)):
            # i = 3
            idx = test_undesired_index[i]
            if idx in idx_instances_evaluated:
                continue
            x_jce_pd = data.jce_test_undesired_pd.loc[idx]
            x_test_pd = data.test_pd.loc[idx]
            x_feat_value = x_test_pd[feat]
            if feat_unique_val_dict[x_feat_value] >= instances_per_feature_val:
                continue
            if all(feat_unique_val_dict.values()) >= instances_per_feature_val:
                break
            else:
                print(f'---------------------------')
                print(f'     Dataset: {data_str}')
                print(f'  Test instance number: {i}')
                print(f'  Idx number: {idx}')
                print(f'  Feat: {feat}')
                print(f'  Feat value: {x_test_pd[feat]}')
                print(f'  Instances feat value: {feat_unique_val_dict[x_feat_value]}')
                print(f'---------------------------')
                feat_unique_val_dict[x_feat_value] = feat_unique_val_dict[x_feat_value] + 1
                idx_instances_evaluated.extend([idx])
                x_target = data.test_undesired_target[i]
                x_jce_np = x_jce_pd.to_numpy()
                x_carla_pd = data.test_undesired_pd.loc[idx].to_frame().T
                x_label = model.jce_sel.predict(x_jce_np.reshape(1,-1))
                data.add_sorted_train_data(x_jce_pd)
                cf_evaluator.add_specific_x_data(idx,x_jce_np,x_test_pd,x_label,x_target,data)
                cf_evaluator.evaluate_cf_models(x_jce_np,x_label,data,model,epsilon_ft,carla_model,x_carla_pd)

                # if 'mace' in models_to_run:
                #     mace_cf_pd = data.mace_df.loc[idx].to_frame().T
                #     if isinstance(mace_cf_pd, pd.Series) or isinstance(mace_cf_pd, pd.DataFrame):
                #         if mace_cf_pd.isnull().values.any():
                #             mace_cf = None
                #             print(f'MACE: Could not find feasible CF!')
                #         else:
                #             mace_cf_pd, mace_cf = data.from_mace_to_jce(mace_cf_pd)
                #             mace_cf = mace_cf[0]
                #     elif np.isnan(np.sum(mace_cf_pd)):
                #         mace_cf = None
                #         print(f'MACE: Could not find feasible CF!')
                #     mace_time = data.mace_time.loc[idx]['time']
                #     cf_evaluator.add_specific_cf_data(data,'mace',mace_cf,mace_time,model.jce_sel)
                #     print(f'  MACE (time (s): {np.round_(mace_time,2)})')
                #     print(f'---------------------------')
                    
    print(f'---------------------------')
    print(f'  DONE: {data_str} CF Evaluation')
    print(f'---------------------------')
    save_obj(cf_evaluator,data_str+'_mutability_eval.pkl')

print(f'---------------------------')
print(f'  DONE: All CFs and Datasets')
print(f'---------------------------')
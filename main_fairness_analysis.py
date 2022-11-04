"""
Imports
"""
import warnings
warnings.filterwarnings("ignore")
from eval import Evaluator
from data_model_load import load_model_dataset
import pickle
import numpy as np
import pandas as pd
from carla_adapter import MyOwnDataSet, MyOwnModel
from support import path_here, save_obj, load_obj

datasets = ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law'] # Name of the dataset to be analyzed
methods_to_run = ['nn','mo','rt','cchvae'] #['nn','mo','ft','rt','gs','face','dice','cchvae'] 
step = 0.01                # Step size to change continuous features
train_fraction = 0.7       # Percentage of examples to use for training
n_feat = 50                # Number of examples to generate synthetically per feature
epsilon_ft = 0.01          # Epsilon corresponding to the rate of change in feature tweaking algorithm
seed_int = 54321           # Seed integer value
only_undesired_cf = 1      # Find counterfactuals only for negative (bad) class factuals
perc = 1                   # Percentage of false negative test samples to consider for the counterfactuals search

np.random.seed(seed_int)

for data_str in datasets:

    data, model = load_model_dataset(data_str, train_fraction, seed_int, step, path_here)
    carla_data = MyOwnDataSet(data)
    carla_model = MyOwnModel(carla_data, data, model)
    save_obj(model, data_str+'_fnr_model.pkl')
    save_obj(data, data_str+'_fnr_data.pkl')
    
    for method_str in methods_to_run:
    
        print(f'---------------------------------------')  
        print(f'                    Dataset: {data_str}')
        print(f'                     Method: {method_str}')
        print(f'        Train dataset shape: {data.train_df.shape}')
        print(f'         Test dataset shape: {data.undesired_transformed_test_np.shape}')
        print(f'       model train accuracy: {np.round_(model.sel.score(data.transformed_train_df, data.train_target), 2)}')
        print(f'        model test accuracy: {np.round_(model.sel.score(data.undesired_test_df, data.undesired_test_target), 2)}')
        print(f' CARLA model train accuracy: {np.round_(carla_model._mymodel.score(data.carla_transformed_train_df, data.train_target),2)}')
        print(f'  CARLA model test accuracy: {np.round_(carla_model._mymodel.score(data.carla_transformed_test_df, data.test_target), 2)}')
        print(f'---------------------------------------')
        
        cf_evaluator = Evaluator(data, n_feat, method_str)        
        cf_evaluator.add_fairness_measures(data, model)
        desired_ground_truth_test_df = data.transformed_test_df.loc[data.test_target != data.undesired_class]
        desired_ground_truth_target = data.test_target[data.test_target != data.undesired_class]
        predicted_label_desired_ground_truth_test_df = model.sel.predict(desired_ground_truth_test_df)
        false_undesired_test_df = desired_ground_truth_test_df.loc[predicted_label_desired_ground_truth_test_df == data.undesired_class]
        false_undesired_target = desired_ground_truth_target[predicted_label_desired_ground_truth_test_df == data.undesired_class]
        for i in range(int(len(false_undesired_test_df)*perc)):
            idx = false_undesired_test_df.index.tolist()[i]
            x_instance = false_undesired_test_df.loc[idx]
            x_target = false_undesired_target[i]
            print(f'---------------------------')
            print(f'     Dataset: {data_str}')
            print(f'      Method: {method_str}')
            print(f'  Test instance number: {i}')
            print(f'  Idx number: {idx}')
            print(f'  Total instances: {int(len(false_undesired_test_df)*perc)}')
            print(f'---------------------------')
            x_np = x_instance.to_numpy()
            x_carla_df = data.test_df.loc[idx].to_frame().T
            x_label = model.sel.predict(x_np.reshape(1,-1))
            data.add_sorted_train_data(x_instance)
            cf_evaluator.add_specific_x_data(idx, x_np, x_carla_df, x_label, x_target, data)
            cf_evaluator.evaluate_cf_models(x_np, x_label, data, model, epsilon_ft, carla_model, x_carla_df)
                        
        print(f'---------------------------')
        print(f'  DONE: {data_str} CF Evaluation')
        print(f'---------------------------')
        save_obj(cf_evaluator, f'{data_str}_{method_str}_fnr_eval.pkl')

print(f'---------------------------')
print(f'  DONE: All CFs and Datasets')
print(f'---------------------------')
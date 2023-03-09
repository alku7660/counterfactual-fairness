"""
Imports
"""
import warnings
warnings.filterwarnings("ignore")
from data_constructor import load_dataset
from model_constructor import Model
from ioi_constructor import IOI
from counterfactual_constructor import Counterfactual
from evaluator_constructor import Evaluator
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from support import path_here, save_obj
import time

datasets = ['german','dutch'] # ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law']
methods_to_run = ['nn','mo','cchvae','ijuice'] # ['nn','mo','ft','rt','gs','face','dice','cchvae'] 
step = 0.01                # Step size to change continuous features
train_fraction = 0.7       # Percentage of examples to use for training
n_feat = 50                # Number of examples to generate synthetically per feature
epsilon_ft = 0.01          # Epsilon corresponding to the rate of change in feature tweaking algorithm
seed_int = 54321           # Seed integer value
only_undesired_cf = 1      # Find counterfactuals only for negative (bad) class factuals
np.random.seed(seed_int)

if __name__=='__main__':

    for data_str in datasets:

        data = load_dataset(data_str, train_fraction, seed_int, step)
        model = Model(data)
        data.undesired_test(model)
        print(f'---------------------------------------')  
        print(f'                    Dataset: {data_str}')
        print(f'                     Method: {method_str}')
        print(f'        Train dataset shape: {data.train_df.shape}')
        print(f'         Test dataset shape: {data.false_undesired_test_df.shape}')
        print(f'       model train accuracy: {np.round_(f1_score(model.model.predict(data.transformed_train_df), data.train_target), 2)}')
        print(f'        model test accuracy: {np.round_(f1_score(model.model.predict(data.transformed_test_df), data.test_target), 2)}')
        print(f'---------------------------------------')
        
        for method_str in methods_to_run:
            num_instances = 3          # Number of false negative test samples to consider for the counterfactuals search
            cf_evaluator = Evaluator(data, n_feat, method_str)        
            cf_evaluator.add_fairness_measures(data, model)
            cf_evaluator.add_fnr_data(data)
            num_instances = num_instances if int(len(data.false_undesired_test_df)) > num_instances else int(len(data.false_undesired_test_df))
            for i in range(num_instances):
                
                ioi = IOI(i, data, model, type='euclidean')
                print(f'---------------------------')
                print(f'     Dataset: {data_str}')
                print(f'      Method: {method_str}')
                print(f'  Idx number: {ioi.idx}')
                print(f'  Test instance number: {i+1}')
                print(f'  Total instances: {num_instances}')
                print(f'---------------------------')
                counterfactual = Counterfactual(data, model, method_str, ioi, type='euclidean')
                cf_evaluator.add_specific_x_data(ioi)
                cf_evaluator.add_specific_cf_data(counterfactual)
                # cf_evaluator.add_specific_x_data(idx, x_np, carla_x_np, x_original_df, x_label, carla_x_label, x_target)
                
                """
                Main function: Find CF for all FN
                """
                # cf_evaluator.evaluate_cf_models(idx, data, model, epsilon_ft, carla_model, cchvae_model=cchvae_model, cchvae_model_time=cchvae_model_time)
                # cf_evaluator.evaluate_cf_models(idx, data, model, epsilon_ft)
                
            """
            Additional functions: Find group counterfactuals and clusters counterfactuals
            """
            # cf_evaluator.add_clusters()
            # cf_evaluator.prepare_groups_clusters_analysis()
            # cf_evaluator.add_clusters_cf(data, model, carla_model, cchvae_model=cchvae_model, cchvae_model_time=cchvae_model_time)
            # cf_evaluator.add_clusters_cf(data, model)
            # cf_evaluator.add_groups_cf(data, model)
                            
            print(f'---------------------------')
            print(f'  DONE: {data_str} CF Evaluation')
            print(f'---------------------------')
            save_obj(cf_evaluator, f'{data_str}_{method_str}_eval.pkl')

    print(f'---------------------------')
    print(f'  DONE: All CFs and Datasets')
    print(f'---------------------------')
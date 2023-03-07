"""
Imports
"""
import warnings
warnings.filterwarnings("ignore")
# from data_model_load import load_model_dataset
from data_constructor import load_dataset
from model_constructor import Model
from ioi_constructor import IOI
from counterfactual_constructor import Counterfactual
from evaluator_constructor import Evaluator
import numpy as np
import pandas as pd
# from carla_adapter import MyOwnDataSet, MyOwnModel
# from carla.recourse_methods import CCHVAE
from sklearn.metrics import f1_score
from support import path_here, save_obj
import time

datasets = ['adult','kdd_census','german','dutch'] # ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law']
methods_to_run = ['nn','mo','cchvae','ijuice'] # ['nn','mo','ft','rt','gs','face','dice','cchvae'] 
step = 0.01                # Step size to change continuous features
train_fraction = 0.7       # Percentage of examples to use for training
n_feat = 50                # Number of examples to generate synthetically per feature
epsilon_ft = 0.01          # Epsilon corresponding to the rate of change in feature tweaking algorithm
seed_int = 54321           # Seed integer value
only_undesired_cf = 1      # Find counterfactuals only for negative (bad) class factuals
perc = 0.1                   # Percentage of false negative test samples to consider for the counterfactuals search
np.random.seed(seed_int)

if __name__=='__main__':

    for data_str in datasets:

        # data, model = load_model_dataset(data_str, train_fraction, seed_int, step, path_here)
        # carla_data = MyOwnDataSet(data)
        # carla_model = MyOwnModel(carla_data, data, model)
        data = load_dataset(data_str, train_fraction, seed_int, step)
        model = Model(data)
        data.undesired_test(model)
        
        for method_str in methods_to_run:
        
            print(f'---------------------------------------')  
            print(f'                    Dataset: {data_str}')
            print(f'                     Method: {method_str}')
            print(f'        Train dataset shape: {data.train_df.shape}')
            print(f'         Test dataset shape: {data.false_undesired_test_df.shape}')
            print(f'       model train accuracy: {np.round_(f1_score(model.model.predict(data.transformed_train_df), data.train_target), 2)}')
            print(f'        model test accuracy: {np.round_(f1_score(model.model.predict(data.transformed_test_df), data.test_target), 2)}')
            # print(f' CARLA model train accuracy: {np.round_(carla_model._mymodel.score(data.carla_transformed_train_df, data.train_target),2)}')
            # print(f'  CARLA model test accuracy: {np.round_(carla_model._mymodel.score(data.carla_transformed_test_df, data.test_target), 2)}')
            print(f'---------------------------------------')
            
            # cchvae_model_start_time = time.time()
            # if method_str == 'cchvae':
            #     dict_cchvae = {'data_name':data.name, 'p_norm':2, 'vae_params':{'layers':[len(carla_model.feature_input_order), int(len(carla_model.feature_input_order)/2)]}}
            #     cchvae_model = CCHVAE(carla_model, dict_cchvae)
            # else:
            #     cchvae_model = None
            # cchvae_model_end_time = time.time()
            # cchvae_model_time = cchvae_model_end_time - cchvae_model_start_time
            cf_evaluator = Evaluator(data, n_feat, method_str)        
            cf_evaluator.add_fairness_measures(data, model)
            # cf_evaluator.add_fnr_data(desired_ground_truth_test_df, false_undesired_test_df, transformed_false_undesired_test_df)
            cf_evaluator.add_fnr_data(data)
            
            for i in range(int(len(data.false_undesired_test_df)*perc)):
                
                ioi = IOI(i, data, model, type='euclidean')
                print(f'---------------------------')
                print(f'     Dataset: {data_str}')
                print(f'      Method: {method_str}')
                print(f'  Idx number: {ioi.idx}')
                print(f'  Test instance number: {i+1}')
                print(f'  Total instances: {int(len(data.false_undesired_test_df)*perc)}')
                print(f'---------------------------')
                # carla_x_transformed_df = data.carla_transformed_test_df.loc[idx].to_frame().T
                # carla_x_np = carla_x_transformed_df.to_numpy()
                # x_original_df = data.test_df.loc[idx].to_frame().T
                # x_transformed_df = data.transformed_false_undesired_test_df.loc[idx]
                # x_np = x_transformed_df.to_numpy()
                # x_target = data.false_undesired_target[i]
                # x_label = model.model.predict(x_np.reshape(1, -1))
                # carla_x_label = model.carla_sel.predict(carla_x_np.reshape(1, -1))
                # data.add_sorted_train_data(x_transformed_df)
                # cf_evaluator.add_specific_x_data(idx, x_np, x_original_df, x_label, x_target)
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
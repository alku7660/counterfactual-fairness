"""
Imports
horrible
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

datasets = ['bank'] # Name of the dataset to be analyzed ['compass','credit','adult','german','kdd_census']
models_to_run = ['nn','mutable-nn','mo','mutable-mo','rt','mutable-rt'] #['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice']
step = 0.01                # Step size to change continuous features
train_fraction = 0.7       # Percentage of examples to use for training
n_feat = 50                # Number of examples to generate synthetically per feature
k = 50                     # Number of training dataset neighbors to consider for the correlation matrix calculation for the CIJN method
epsilon_ft = 0.01          # Epsilon corresponding to the rate of change in feature tweaking algorithm
seed_int = 54321           # Seed integer value
only_undesired_cf = 1      # Find counterfactuals only for negative (bad) class factuals
perc = 1             # Percentage of test samples to consider for the counterfactuals search
instances_per_protected_class = 5

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
            idx = test_undesired_index[i]
            if idx in idx_instances_evaluated:
                continue
            x_jce_pd = data.jce_test_undesired_pd.loc[idx]
            x_test_pd = data.test_pd.loc[idx]
            x_feat_value = x_test_pd[feat]
            if feat_unique_val_dict[x_feat_value] >= instances_per_protected_class:
                continue
            if all(feat_unique_val_dict.values()) >= instances_per_protected_class:
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
                cf_evaluator.evaluate_cf_models(x_jce_np,x_label,data,model,epsilon_ft)
                # if 'nn' in models_to_run:
                #     nn_cf, nn_time = nn(x_jce_np,x_label,data)
                #     cf_evaluator.add_specific_cf_data(data,'nn',nn_cf,nn_time,model.jce_sel,nn_cf,1,1)
                #     print(f'  NN (time (s): {np.round_(nn_time,2)})')
                #     print(f'---------------------------')

                # if 'mutable-nn' in models_to_run:
                #     mutable_nn_cf, mutable_nn_time = nn(x_jce_np,x_label,data,mutability_check=False)
                #     cf_evaluator.add_specific_cf_data(data,'mutable-nn',mutable_nn_cf,mutable_nn_time,model.jce_sel,mutable_nn_cf,1,1)
                #     print(f'  Mutable NN (time (s): {np.round_(mutable_nn_time,2)})')
                #     print(f'---------------------------')

                # if 'mo' in models_to_run:
                #     mo_cf, mo_time = min_obs(x_jce_np,x_label,data)
                #     cf_evaluator.add_specific_cf_data(data,'mo',mo_cf,mo_time,model.jce_sel)
                #     print(f'  MO (time (s): {np.round_(mo_time,2)})')
                #     print(f'---------------------------')

                # if 'mutable-mo' in models_to_run:
                #     mutable_mo_cf, mutable_mo_time = min_obs(x_jce_np,x_label,data,mutability_check=False)
                #     cf_evaluator.add_specific_cf_data(data,'mutable-mo',mutable_mo_cf,mutable_mo_time,model.jce_sel)
                #     print(f'  Mutable MO (time (s): {np.round_(mutable_mo_time,2)})')
                #     print(f'---------------------------')

                # if 'ft' in models_to_run:
                #     ft_cf, ft_time = feat_tweak(x_jce_np,model.jce_rf,epsilon_ft)
                #     cf_evaluator.add_specific_cf_data(data,'ft',ft_cf,ft_time,model.jce_rf)
                #     print(f'  FT (time (s): {np.round_(ft_time,2)})')
                #     print(f'---------------------------')

                # if 'rt' in models_to_run:
                #     rt_cf, rt_time = rf_tweak(x_jce_np,x_label,model.jce_rf,data)
                #     cf_evaluator.add_specific_cf_data(data,'rt',rt_cf,rt_time,model.jce_rf)
                #     print(f'  RT (time (s): {np.round_(rt_time,2)})')
                #     print(f'---------------------------')

                # if 'mutable-rt' in models_to_run:
                #     mutable_rt_cf, mutable_rt_time = rf_tweak(x_jce_np,x_label,model.jce_rf,data,feasibility_check=True,mutability_check=False)
                #     cf_evaluator.add_specific_cf_data(data,'mutable-rt',mutable_rt_cf,mutable_rt_time,model.jce_rf)
                #     print(f'  Mutable RT (time (s): {np.round_(mutable_rt_time,2)})')
                #     print(f'---------------------------')

                # if 'gs' in models_to_run:
                #     start_time = time.time()
                #     gs_model = GrowingSpheres(carla_model)
                #     gs_cf_pd = gs_model.get_counterfactuals(x_carla_pd)
                #     if isinstance(gs_cf_pd, pd.Series) or isinstance(gs_cf_pd, pd.DataFrame):
                #         if gs_cf_pd.isnull().values.any():
                #             gs_cf = None
                #             print(f'GS: Could not find feasible CF!')
                #         else:
                #             gs_cf_pd = data.from_carla_to_jce(gs_cf_pd)
                #             gs_cf = np.array(gs_cf_pd)[0]
                #     elif np.isnan(np.sum(gs_cf_pd)):
                #         gs_cf = None
                #         print(f'GS: Could not find feasible CF!')
                #     end_time = time.time() 
                #     gs_time = end_time - start_time
                #     cf_evaluator.add_specific_cf_data(data,'gs',gs_cf,gs_time,model.jce_sel)
                #     print(f'  GS (time (s): {np.round_(gs_time,2)})')
                #     print(f'---------------------------')

                # if 'face' in models_to_run:
                #     start_time = time.time()
                #     face_model = Face(carla_model, {'mode':'knn','fraction':0.2})
                #     face_knn_cf_pd = face_model.get_counterfactuals(x_carla_pd)
                #     if isinstance(face_knn_cf_pd, pd.Series) or isinstance(face_knn_cf_pd, pd.DataFrame):
                #         if face_knn_cf_pd.isnull().values.any():
                #             face_knn_cf = None
                #             print(f'FACE-knn: Could not find feasible CF!')
                #         else:
                #             face_knn_cf_pd = data.from_carla_to_jce(face_knn_cf_pd)
                #             face_knn_cf = np.array(face_knn_cf_pd)[0]
                #     elif np.isnan(np.sum(face_knn_cf_pd)):
                #         face_knn_cf = None
                #         print(f'FACE-knn: Could not find feasible CF!')
                #     end_time = time.time() 
                #     face_time = end_time - start_time
                #     cf_evaluator.add_specific_cf_data(data,'face_knn',face_knn_cf,face_time,model.jce_sel)
                #     print(f'  FACE (time (s): {np.round_(face_time,2)})')
                #     print(f'---------------------------')

                # if 'dice' in models_to_run:
                #     start_time = time.time()
                #     dice_model = Dice(carla_model, {'desired_class': int(1-data.undesired_class)})
                #     dice_cf_pd = dice_model.get_counterfactuals(x_carla_pd)
                #     if isinstance(dice_cf_pd, pd.Series) or isinstance(dice_cf_pd, pd.DataFrame):
                #         if dice_cf_pd.isnull().values.any():
                #             dice_cf = None
                #             print(f'DICE: Could not find feasible CF!')
                #         else:
                #             dice_cf_pd = data.from_carla_to_jce(dice_cf_pd)
                #             dice_cf = np.array(dice_cf_pd)[0]
                #     elif dice_cf_pd is None:
                #         dice_cf = None
                #         print(f'DICE: Could not find feasible CF!')
                #     elif dice_cf_pd is not None:
                #         if np.isnan(np.sum(dice_cf_pd)):
                #             dice_cf = None
                #             dice_cf_pd = None
                #             print(f'DICE: Could not find feasible CF!')
                #     end_time = time.time() 
                #     dice_time = end_time - start_time
                #     cf_evaluator.add_specific_cf_data(data,'dice',dice_cf,dice_time,model.jce_sel)
                #     print(f'  DICE (time (s): {np.round_(dice_time,2)})')
                #     print(f'---------------------------')

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

                # if 'cchvae' in models_to_run:
                #     data_with_target = copy.deepcopy(data.train_pd)
                #     data_with_target[data.label_str[0]] = data.train_target
                #     start_time = time.time()
                #     cchvae_model = CCHVAE(carla_model, {'data_name': data_str,'p_norm':2,'vae_params':{'layers':[len(carla_model.feature_input_order),int(len(carla_model.feature_input_order)/2)]}}, data_with_target)
                #     cchvae_cf_pd = cchvae_model.get_counterfactuals(x_carla_pd)
                #     if isinstance(cchvae_cf_pd, pd.Series) or isinstance(cchvae_cf_pd, pd.DataFrame):
                #         if cchvae_cf_pd.isnull().values.any():
                #             cchvae_cf = None
                #             print(f'CCHVAE: Could not find feasible CF!')
                #         else:
                #             cchvae_cf_pd = data.from_carla_to_jce(cchvae_cf_pd)
                #             cchvae_cf = np.array(cchvae_cf_pd)[0]
                #     elif cchvae_cf_pd is None:
                #         cchvae_cf = None
                #         print(f'CCHVAE: Could not find feasible CF!')
                #     elif cchvae_cf_pd is not None:
                #         if np.isnan(np.sum(cchvae_cf_pd)):
                #             cchvae_cf = None
                #             cchvae_cf_pd = None
                #             print(f'CCHVAE: Could not find feasible CF!')
                #     end_time = time.time() 
                #     cchvae_time = end_time - start_time
                #     cf_evaluator.add_specific_cf_data(data,'cchvae',cchvae_cf,cchvae_time,model.jce_sel)
                #     print(f'  CCHVAE (time (s): {np.round_(cchvae_time,2)})')
                #     print(f'---------------------------')

                # if 'juice' in models_to_run:
                #     jce_prox_cf, instance_jce_prox, just_jce_prox, found_justifiable_jce_prox, jce_prox_time = JUICE(x_jce_np,x_label,data,model.jce_sel,'proximity') # Refer to jce.py for details
                #     cf_evaluator.add_specific_cf_data(data,'jce_prox',jce_prox_cf,jce_prox_time,model.jce_sel,instance_jce_prox,just_jce_prox,found_justifiable_jce_prox)
                #     print(f'  P-JCE (time (s): {np.round_(jce_prox_time,2)})')
                #     print(f'---------------------------')

                #     jce_spar_cf, instance_jce_spar, just_jce_spar, found_justifiable_jce_spar, jce_spar_time = JUICE(x_jce_np,x_label,data,model.jce_sel,'sparsity') # Refer to jce.py for details
                #     sparsity_changed_feat, jce_spar_cf = jce_spar_cf[1], jce_spar_cf[0]
                #     cf_evaluator.add_specific_cf_data(data,'jce_spar',jce_spar_cf,jce_spar_time,model.jce_sel,instance_jce_spar,just_jce_spar,found_justifiable_jce_spar)
                #     print(f'  S-JCE (time (s): {np.round_(jce_spar_time,2)})')
                #     print(f'---------------------------')
                
                # if 'mutable-juice' in models_to_run:
                #     mutable_jce_prox_cf, mutable_instance_jce_prox, mutable_just_jce_prox, mutable_found_justifiable_jce_prox, mutable_jce_prox_time = JUICE(x_jce_np,x_label,data,model.jce_sel,'proximity',mutability_check=False) # Refer to jce.py for details
                #     cf_evaluator.add_specific_cf_data(data,'mutable-jce_prox',mutable_jce_prox_cf,mutable_jce_prox_time,model.jce_sel,mutable_instance_jce_prox,mutable_just_jce_prox,mutable_found_justifiable_jce_prox)
                #     print(f'  Mutable P-JCE (time (s): {np.round_(mutable_jce_prox_time,2)})')
                #     print(f'---------------------------')

                #     mutable_jce_spar_cf, mutable_instance_jce_spar, mutable_just_jce_spar, mutable_found_justifiable_jce_spar, mutable_jce_spar_time = JUICE(x_jce_np,x_label,data,model.jce_sel,'sparsity',mutability_check=False) # Refer to jce.py for details
                #     mutable_sparsity_changed_feat, mutable_jce_spar_cf = mutable_jce_spar_cf[1], mutable_jce_spar_cf[0]
                #     cf_evaluator.add_specific_cf_data(data,'mutable-jce_spar',mutable_jce_spar_cf,mutable_jce_spar_time,model.jce_sel,mutable_instance_jce_spar,mutable_just_jce_spar,mutable_found_justifiable_jce_spar)
                #     print(f'  Mutable S-JCE (time (s): {np.round_(mutable_jce_spar_time,2)})')
                #     print(f'---------------------------')
                    
    print(f'---------------------------')
    print(f'  DONE: {data_str} CF Evaluation')
    print(f'---------------------------')
    save_obj(cf_evaluator,data_str+'_mutability_eval.pkl')

print(f'---------------------------')
print(f'  DONE: All CFs and Datasets')
print(f'---------------------------')
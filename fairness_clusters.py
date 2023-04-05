"""
Imports
"""
import warnings
warnings.filterwarnings("ignore")
from data_constructor import load_dataset
from model_constructor import Model
from ioi_constructor import IOI
from clusters_constructor import Clusters
from centroid_constructor import Centroid
from cluster_counterfactual_constructor import Counterfactual
from evaluator_constructor import Evaluator
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from support import path_here, save_obj
import time

datasets = ['german'] # ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law']
methods_to_run = ['fijuice'] # ['nn','mo','ft','rt','gs','face','dice','cchvae','juice','ijuice']
step = 0.01                # Step size to change continuous features
train_fraction = 0.7       # Percentage of examples to use for training
n_feat = 50                # Number of examples to generate synthetically per feature
epsilon_ft = 0.01          # Epsilon corresponding to the rate of change in feature tweaking algorithm
seed_int = 54321           # Seed integer value
only_undesired_cf = 1      # Find counterfactuals only for negative (bad) class factuals
clustering_metric = 'complete' # Clustering metric used
dist = 'L1_L0'
lagrange = 0.0
np.random.seed(seed_int)

if __name__=='__main__':
    for data_str in datasets:
        data = load_dataset(data_str, train_fraction, seed_int, step)
        model = Model(data)
        data.undesired_test(model)
        clusters_obj = Clusters(data, model, metric=clustering_metric)
        
        print(f'---------------------------------------')
        print(f'                    Dataset: {data_str}')
        print(f'        Train dataset shape: {data.train_df.shape}')
        print(f'         Test dataset shape: {data.false_undesired_test_df.shape}')
        print(f'       model train accuracy: {np.round_(f1_score(model.model.predict(data.transformed_train_df), data.train_target), 2)}')
        print(f'        model test accuracy: {np.round_(f1_score(model.model.predict(data.transformed_test_df), data.test_target), 2)}')
        print(f'                   Lagrange: {lagrange}')
        print(f'---------------------------------------')
        cf_evaluator = Evaluator(data, n_feat, methods_to_run[0])
        cf_evaluator.add_fairness_measures(data, model)
        cf_evaluator.add_fnr_data(data)
        cf_evaluator.add_cluster_data(clusters_obj)
        
        print(f'---------------------------')
        print(f'    Dataset: {data_str}')
        print(f'     Method: {methods_to_run[0]}')
        print(f'---------------------------')
        counterfactual = Counterfactual(data, model, methods_to_run[0], clusters_obj, type=dist, lagrange=lagrange, t=100, k=1)
        cf_evaluator.add_cf_data(counterfactual)

        print(f'---------------------------')
        print(f'  DONE: {data_str} CF Evaluation')
        print(f'---------------------------')
        save_obj(cf_evaluator, f'{data_str}_fijuice_cluster_eval.pkl')

    print(f'---------------------------')
    print(f'  DONE: All CFs and Datasets')
    print(f'---------------------------')
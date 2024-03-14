"""
Imports
"""
import warnings
warnings.filterwarnings("ignore")
from data_constructor import load_dataset
from model_constructor import Model
from clusters_constructor import Clusters
from cluster_counterfactual_constructor import Counterfactual
from evaluator_constructor import Evaluator
import numpy as np
from sklearn.metrics import f1_score
from support import save_obj
import os
import time

# LIST OF DATASETS TO RUN: 'synthetic_athlete','compass','german','student','dutch','oulad','adult','credit'
datasets_zeus = ['dutch']
datasets_home = ['dutch']
datasets_thor = ['dutch']
# Done for CounterFair dist: 'synthetic_athlete','compass','german','student'

print(os.getcwd())

if 'dsv' in os.getcwd():
    if '/data0/' in os.getcwd():
        print('Selected Datasets and cores for Zeus run')
        datasets = datasets_zeus
    else:
        print('Selected Datasets and cores for Thor run')
        datasets = datasets_thor
else:
    print('Selected Datasets and cores for Local run')
    datasets = datasets_home
datasets = ['synthetic_athlete','compass','german']
methods_to_run = ['BIGRACE_dist','ARES','FACTS'] # ['BIGRACE_dist','BIGRACE_l','BIGRACE_e','BIGRACE_dev_dist','BIGRACE_dev_like','BIGRACE_dev_eff','ARES','FACTS']
step = 0.01                # Step size to change continuous features
train_fraction = 0.7       # Percentage of examples to use for training
n_feat = 50                # Number of examples to generate synthetically per feature
epsilon_ft = 0.01          # Epsilon corresponding to the rate of change in feature tweaking algorithm
seed_int = 54321           # Seed integer value
only_undesired_cf = 1      # Find counterfactuals only for negative (bad) class factuals
clustering_metric = 'complete' # Clustering metric used
dist = 'L1_L0'
lagranges = [0.5]  # [0.5] [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
likelihood_factors = [0.5] # [0.5] [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] This is used to calculate a minimum rho admitted for each CF found
# t = 100 # Number of preselected close NN Training Counterfactuals
# k = 10
weight = 0.5
np.random.seed(seed_int)

def percentage_close_train(dataset):
    """
    Selects the appropriate percentage per dataset for the close CF
    """
    if dataset in ['synthetic_athlete','compass','student','german']:
        percentage_close_train_cf = 1
        continuous_bins = 10
    elif dataset in ['dutch']:
        percentage_close_train_cf = 0.02
        continuous_bins = 3
    elif dataset in ['adult']:
        percentage_close_train_cf = 0.1
        continuous_bins = 5
    elif dataset in ['credit','oulad']:
        percentage_close_train_cf = 0.05
        continuous_bins = 3
    return percentage_close_train_cf, continuous_bins

def support_threshold(dataset):
    """
    Selects the appropriate support threshold
    """
    if dataset in ['compass','synthetic_athlete','german','adult']:
        support_th = 0.05
    elif dataset in ['dutch']:
        support_th = 0.1
    elif dataset in ['student']:
        support_th = 0.3
    return support_th

def select_parameters(method, weight):
    """
    Selects the parameters according to the type of BIGRACE method
    """
    if method == 'BIGRACE_dist':
        alpha, dev, eff = weight, False, False
    elif method == 'BIGRACE_dev_dist':
        alpha, dev, eff = weight, True, False
    elif method == 'BIGRACE_e':
        alpha, dev, eff = weight, False, True
    else:
        alpha, dev, eff = 0.0, False, False
    return alpha, dev, eff

if __name__=='__main__':
    for data_str in datasets:
        percentage_close_train_cf, continuous_bins = percentage_close_train(data_str)
        support_th = support_threshold(data_str)
        data = load_dataset(data_str, train_fraction, seed_int, step)
        model = Model(data)
        data.undesired_test(model)
        print(f'---------------------------------------')
        print(f'                    Dataset: {data_str}')
        print(f'        Train dataset shape: {data.train_df.shape}')
        print(f'         Test dataset shape: {data.false_undesired_test_df.shape}')
        print(f'       model train accuracy: {np.round_(f1_score(model.model.predict(data.transformed_train_df), data.train_target), 2)}')
        print(f'        model test accuracy: {np.round_(f1_score(model.model.predict(data.transformed_test_df), data.test_target), 2)}')
        print(f'---------------------------------------')
        clusters_obj = Clusters(data, model, metric=clustering_metric)
        for method in methods_to_run:
            start_time = time.time()
            cf_evaluator = Evaluator(data, n_feat, method, clusters_obj)
            if 'BIGRACE' in method:
                cf_evaluator.add_fairness_measures(data, model)
                cf_evaluator.add_fnr_data(data)
                alpha, dev, eff = select_parameters(method, weight)
                # counterfactual = Counterfactual(data, model, method, clusters_obj, alpha, dev, eff type=dist, percentage_close_train_cf=percentage_close_train_cf, support_th=support_th)
                counterfactual = Counterfactual(data, model, method, alpha, dev, eff, type=dist, percentage_close_train_cf=percentage_close_train_cf, support_th=support_th, continuous_bins=continuous_bins)
                cf_evaluator.add_cf_data(counterfactual)
            elif method == 'ARES':
                graph_obj = None
                alpha, dev, eff = select_parameters(method, weight)
                counterfactual = Counterfactual(data, model, method, alpha, dev, eff, type=dist, percentage_close_train_cf=percentage_close_train_cf, support_th=support_th, continuous_bins=continuous_bins, cluster=clusters_obj)
                cf_evaluator.add_cf_data_ares(counterfactual)
            elif method == 'FACTS':
                graph_obj = None
                alpha, dev, eff = select_parameters(method, weight)
                counterfactual = Counterfactual(data, model, method, alpha, dev, eff, type=dist, percentage_close_train_cf=percentage_close_train_cf, support_th=support_th, continuous_bins=continuous_bins, cluster=clusters_obj)
                cf_evaluator.add_cf_data_facts(counterfactual)
            end_time = time.time()
            print(f'---------------------------')
            print(f'  DONE: {data_str}, method: {method}, time: {np.round(end_time - start_time, 2)}')
            print(f'---------------------------')
            if dev == False and eff == False:
                save_obj(cf_evaluator, f'{data_str}_{method}_alpha_{alpha}_eval.pkl')
            elif dev == True:
                save_obj(cf_evaluator, f'{data_str}_{method}_dev_eval.pkl')
            elif eff == True:
                save_obj(cf_evaluator, f'{data_str}_{method}_eff_eval.pkl')
            elif dev == False and eff == False and alpha == 0.0:
                save_obj(cf_evaluator, f'{data_str}_{method}_eval.pkl')
    print(f'---------------------------')
    print(f'  DONE: All CFs and Datasets')
    print(f'---------------------------')
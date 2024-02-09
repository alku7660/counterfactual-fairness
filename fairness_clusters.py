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
import time

datasets = ['compass'] # 'compass','synthetic_athlete','german','oulad','bank','student','law','credit','dutch','adult' # 'diabetes','kdd_census'
methods_to_run = ['ARES'] # ['BIGRACE_dist','BIGRACE_l','BIGRACE_e','BIGRACE_dev_dist','BIGRACE_dev_like','BIGRACE_dev_eff','ARES','FACTS']
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
major_weight, minor_weight = 1.0, 0.0
np.random.seed(seed_int)

def percentage_close_train(dataset):
    """
    Selects the appropriate percentage per dataset for the close CF
    """
    if dataset in ['german','compass','synthetic_athlete','diabetes']:
        percentage_close_train_cf = 0.1
    elif dataset in ['bank','kdd_census','student']:
        percentage_close_train_cf = 0.1
    elif dataset in ['oulad','adult','credit','dutch']:
        percentage_close_train_cf = 0.01
    return percentage_close_train_cf

def support_threshold(dataset):
    """
    Selects the appropriate support threshold
    """
    if dataset in ['compass','synthetic_athlete','synthetic_disease','credit','adult','kdd_census','diabetes']:
        support_th = 0.05
    elif dataset in ['german']:
        support_th = 0.05
    elif dataset in ['dutch','oulad','law']:
        support_th = 0.1
    elif dataset in ['bank','law']:
        support_th = 0.25
    elif dataset in ['student']:
        support_th = 0.4
    return support_th

def select_parameters(method):
    """
    Selects the parameters according to the type of BIGRACE method
    """
    if method == 'BIGRACE_dist':
        alpha, beta, gamma, delta1, delta2, delta3 = major_weight, minor_weight, minor_weight, minor_weight, minor_weight, minor_weight
    elif method == 'BIGRACE_l':
        alpha, beta, gamma, delta1, delta2, delta3 = minor_weight, major_weight, minor_weight, minor_weight, minor_weight, minor_weight
    elif method == 'BIGRACE_e':
        alpha, beta, gamma, delta1, delta2, delta3 = minor_weight, minor_weight, major_weight, minor_weight, minor_weight, minor_weight
    elif method == 'BIGRACE_dev_dist':
        alpha, beta, gamma, delta1, delta2, delta3 = minor_weight, minor_weight, minor_weight, major_weight, minor_weight, minor_weight
    elif method == 'BIGRACE_dev_like':
        alpha, beta, gamma, delta1, delta2, delta3 = minor_weight, minor_weight, minor_weight, minor_weight, major_weight, minor_weight
    elif method == 'BIGRACE_dev_eff':
        alpha, beta, gamma, delta1, delta2, delta3 = minor_weight, minor_weight, minor_weight, minor_weight, minor_weight, major_weight
    else:
        alpha, beta, gamma, delta1, delta2, delta3 = minor_weight, minor_weight, minor_weight, minor_weight, minor_weight, minor_weight
    return alpha, beta, gamma, delta1, delta2, delta3

if __name__=='__main__':
    for data_str in datasets:
        percentage_close_train_cf = percentage_close_train(data_str)
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
                print('Before add_fairness_measures')
                cf_evaluator.add_fairness_measures(data, model)
                print('After add_fairness_measures Before add_fnr_data')
                cf_evaluator.add_fnr_data(data)
                print('After add_fnr_data Before Counterfactual creation')
                alpha, beta, gamma, delta1, delta2, delta3 = select_parameters(method)
                # counterfactual = Counterfactual(data, model, method, clusters_obj, alpha, beta, gamma, delta1, delta2, delta3, type=dist, percentage_close_train_cf=percentage_close_train_cf, support_th=support_th)
                counterfactual = Counterfactual(data, model, method, alpha, beta, gamma, delta1, delta2, delta3, type=dist, percentage_close_train_cf=percentage_close_train_cf, support_th=support_th)
                cf_evaluator.add_cf_data(counterfactual)
            elif method == 'ARES':
                graph_obj = None
                alpha, beta, gamma, delta1, delta2, delta3 = select_parameters(method)
                counterfactual = Counterfactual(data, model, method, alpha, beta, gamma, delta1, delta2, delta3, type=dist, percentage_close_train_cf=percentage_close_train_cf, support_th=support_th, cluster=clusters_obj)
                cf_evaluator.add_cf_data_ares(counterfactual)
            elif method == 'FACTS':
                graph_obj = None
                alpha, beta, gamma, delta1, delta2, delta3 = select_parameters(method)
                counterfactual = Counterfactual(data, model, method, alpha, beta, gamma, delta1, delta2, delta3, type=dist, percentage_close_train_cf=percentage_close_train_cf, support_th=support_th, cluster=clusters_obj)
                cf_evaluator.add_cf_data_facts(counterfactual)
            end_time = time.time()
            print(f'---------------------------')
            print(f'  DONE: {data_str}, method: {method}, time: {np.round(end_time - start_time, 2)}')
            print(f'---------------------------')
            save_obj(cf_evaluator, f'{data_str}_{method}_cluster_eval.pkl')
    print(f'---------------------------')
    print(f'  DONE: All CFs and Datasets')
    print(f'---------------------------')
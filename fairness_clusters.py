"""
Imports
"""
import warnings
warnings.filterwarnings("ignore")
from data_constructor import load_dataset
from model_constructor import Model
from clusters_constructor import Clusters
from graph_constructor import Graph
from cluster_counterfactual_constructor import Counterfactual
from evaluator_constructor import Evaluator
import numpy as np
from sklearn.metrics import f1_score
from support import save_obj

datasets = ['dutch','compass','synthetic_athlete','heart','student','oulad','bank','law','credit','adult','kdd_census','diabetes','synthetic_disease'] # 'german','dutch','compass','synthetic_athlete','heart','student','oulad','bank','law','credit','adult','kdd_census','diabetes','synthetic_disease'
methods_to_run = ['ARES','FACTS'] # ['FOCE_dist','FOCE_l','FOCE_e','ARES','FACTS']
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
major_weight, minor_weight = 1.00, 0.00
alphas =  [major_weight, minor_weight, minor_weight, minor_weight]
betas = [minor_weight, major_weight, minor_weight, minor_weight]
gammas = [minor_weight, minor_weight, major_weight, minor_weight]
deltas = [minor_weight, minor_weight, minor_weight, major_weight]
np.random.seed(seed_int)

def percentage_close_train(dataset):
    """
    Selects the appropriate percentage per dataset for the close CF
    """
    if dataset in ['german','dutch','compass','synthetic_athlete','synthetic_disease','heart','student']:
        percentage_close_train_cf = 0.05
    elif dataset in ['oulad']:
        percentage_close_train_cf = 0.01
    elif dataset in ['bank','law','credit']:
        percentage_close_train_cf = 0.001
    elif dataset in ['adult','kdd_census','diabetes']:
        percentage_close_train_cf = 0.0001
    return percentage_close_train_cf

def support_threshold(dataset):
    """
    Selects the appropriate support threshold
    """
    if dataset in ['german','compass','synthetic_athlete','synthetic_disease','heart','student','oulad','bank','law','credit','adult','kdd_census','diabetes']:
        support_th = 0.01
    elif dataset in ['dutch']:
        support_th = 0.05
    elif dataset in []:
        support_th = 0.001
    elif dataset in []:
        support_th = 0.0001
    return support_th

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
        if any('FOCE' in x for x in methods_to_run):
            graph_obj = Graph(data, model, clusters_obj, dist, percentage=percentage_close_train_cf)
        else:
            graph_obj = None
        for method in methods_to_run:
            cf_evaluator = Evaluator(data, n_feat, methods_to_run[0], clusters_obj)
            cf_evaluator.add_fairness_measures(data, model)
            cf_evaluator.add_fnr_data(data)
            if method == 'FOCE_dist':
                alpha, beta, gamma, delta = major_weight, minor_weight, minor_weight, minor_weight
            elif method == 'FOCE_l':
                alpha, beta, gamma, delta = minor_weight, major_weight, minor_weight, minor_weight
            elif method == 'FOCE_dev':
                alpha, beta, gamma, delta = minor_weight, minor_weight, major_weight, minor_weight
            elif method == 'FOCE_e':
                alpha, beta, gamma, delta = minor_weight, minor_weight, minor_weight, major_weight
            else:
                alpha, beta, gamma, delta = minor_weight, minor_weight, minor_weight, minor_weight
            counterfactual = Counterfactual(data, model, method, clusters_obj, alpha, beta, gamma, delta, type=dist, graph=graph_obj, support_th=support_th)
            if method == 'ARES':
                cf_evaluator.add_cf_data_ares(counterfactual)
            elif method == 'FACTS':
                cf_evaluator.add_cf_data_facts(counterfactual)
            else:
                cf_evaluator.add_cf_data(counterfactual)
            print(f'---------------------------')
            print(f'  DONE: {data_str}, method: {method}')
            print(f'---------------------------')
            save_obj(cf_evaluator, f'{data_str}_{method}_cluster_eval.pkl')
    print(f'---------------------------')
    print(f'  DONE: All CFs and Datasets')
    print(f'---------------------------')
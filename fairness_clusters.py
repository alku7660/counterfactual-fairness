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

datasets = ['german','dutch','compass','oulad','synthetic_athlete'] # 'german','dutch','compass','oulad','synthetic_athlete' ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law']
methods_to_run = ['fijuice'] # ['nn','mo','ft','rt','gs','face','dice','cchvae','juice','ijuice']
step = 0.01                # Step size to change continuous features
train_fraction = 0.7       # Percentage of examples to use for training
n_feat = 50                # Number of examples to generate synthetically per feature
epsilon_ft = 0.01          # Epsilon corresponding to the rate of change in feature tweaking algorithm
seed_int = 54321           # Seed integer value
only_undesired_cf = 1      # Find counterfactuals only for negative (bad) class factuals
clustering_metric = 'complete' # Clustering metric used
dist = 'L1_L0'
lagranges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
likelihood_factors = [0.2, 0.4, 0.6, 0.8]
np.random.seed(seed_int)

if __name__=='__main__':
    for data_str in datasets:
        data = load_dataset(data_str, train_fraction, seed_int, step)
        model = Model(data)
        data.undesired_test(model)
        clusters_obj = Clusters(data, model, metric=clustering_metric)
        cf_evaluator = Evaluator(data, n_feat, methods_to_run[0], clusters_obj)
        cf_evaluator.add_fairness_measures(data, model)
        cf_evaluator.add_fnr_data(data)
        graph_obj = Graph(data, model, clusters_obj, dist, t=100, k=1)
        for lagrange in lagranges:
            for likelihood_factor in likelihood_factors:
                print(f'---------------------------------------')
                print(f'                    Dataset: {data_str}')
                print(f'        Train dataset shape: {data.train_df.shape}')
                print(f'         Test dataset shape: {data.false_undesired_test_df.shape}')
                print(f'       model train accuracy: {np.round_(f1_score(model.model.predict(data.transformed_train_df), data.train_target), 2)}')
                print(f'        model test accuracy: {np.round_(f1_score(model.model.predict(data.transformed_test_df), data.test_target), 2)}')
                print(f'                   lagrange: {lagrange}')
                print(f'                 Likelihood: {likelihood_factor}')
                print(f'---------------------------------------')
            counterfactual = Counterfactual(data, model, methods_to_run[0], clusters_obj, lagrange, likelihood_factor, type=dist, t=100, k=1, graph=graph_obj)
            cf_evaluator.add_cf_data(counterfactual, lagrange)
            print(f'---------------------------')
            print(f'  DONE: {data_str}, {lagrange} CF Evaluation')
            print(f'---------------------------')
        print(f'---------------------------')
        print(f'  DONE: {data_str}')
        print(f'---------------------------')
        save_obj(cf_evaluator, f'{data_str}_{methods_to_run[0]}_cluster_eval.pkl')

    print(f'---------------------------')
    print(f'  DONE: All CFs and Datasets')
    print(f'---------------------------')
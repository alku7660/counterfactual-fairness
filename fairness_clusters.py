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

datasets = ['german','compass','oulad','synthetic_athlete'] # 'german','dutch','compass','oulad','synthetic_athlete'
methods_to_run = ['fijuice_like_optimize'] # ['nn','mo','ft','rt','gs','face','dice','cchvae','juice','ijuice','fijuice_like_constraint','fijuice_like_optimize','ares']
step = 0.01                # Step size to change continuous features
train_fraction = 0.7       # Percentage of examples to use for training
n_feat = 50                # Number of examples to generate synthetically per feature
epsilon_ft = 0.01          # Epsilon corresponding to the rate of change in feature tweaking algorithm
seed_int = 54321           # Seed integer value
only_undesired_cf = 1      # Find counterfactuals only for negative (bad) class factuals
clustering_metric = 'complete' # Clustering metric used
dist = 'L1_L0'
lagranges = [0.5]  # [0.5] [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
likelihood_factors = [0.0] # [0.5] [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
t = 100 # Number of preselected close NN Training Counterfactuals
k = 10
alphas =  [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
gammas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
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
        # if methods_to_run[0] in ['fijuice_like_constraint','fijuice_like_optimize']:
        graph_obj = Graph(data, model, clusters_obj, dist, t=t, k=k)
        for lagrange in lagranges:
            for likelihood_factor in likelihood_factors:
                for alpha in alphas:
                    for beta in betas:
                        for gamma in gammas:
                            print(f'---------------------------------------')
                            print(f'                    Dataset: {data_str}')
                            print(f'        Train dataset shape: {data.train_df.shape}')
                            print(f'         Test dataset shape: {data.false_undesired_test_df.shape}')
                            print(f'       model train accuracy: {np.round_(f1_score(model.model.predict(data.transformed_train_df), data.train_target), 2)}')
                            print(f'        model test accuracy: {np.round_(f1_score(model.model.predict(data.transformed_test_df), data.test_target), 2)}')
                            print(f'                   lagrange: {lagrange}')
                            print(f'          likelihood factor: {likelihood_factor}')
                            print(f'                      alpha: {alpha}')
                            print(f'                       beta: {beta}')
                            print(f'                      gamma: {gamma}')
                            print(f'---------------------------------------')
                            counterfactual = Counterfactual(data, model, methods_to_run[0], clusters_obj, lagrange, likelihood_factor, alpha, beta, gamma, type=dist, t=100, k=1, graph=graph_obj)
                            cf_evaluator.add_cf_data(counterfactual, lagrange)
                            print(f'---------------------------')
                            print(f'  DONE: {data_str}, lagrange: {lagrange}, likelihood: {likelihood_factor}, alpha: {alpha}, beta: {beta}, gamma: {gamma}')
                print(f'---------------------------')
        # elif methods_to_run[0] == 'ares':
        print(f'---------------------------')
        print(f'  DONE: {data_str}')
        print(f'---------------------------')
        save_obj(cf_evaluator, f'{data_str}_{methods_to_run[0]}_cluster_eval.pkl')
    print(f'---------------------------')
    print(f'  DONE: All CFs and Datasets')
    print(f'---------------------------')
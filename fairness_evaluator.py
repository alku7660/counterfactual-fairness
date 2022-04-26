import numpy as np
import pandas as pd
import pickle
from support import path_here, results_cf_obj_dir, results_cf_plots_dir
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

def load_obj(file_name):
    """
    Method to read an Evaluator object containing the evaluation results for all the instances of a given dataset
    """
    with open(results_cf_obj_dir+file_name, 'rb') as input:
        evaluator_obj = pickle.load(input)
    return evaluator_obj

def extract_x_cd_df(eval_cf_df,eval_x_df):
    """
    Method that joins accuracy, proximity and sparsity from the eval object and outputs the join as a single dataframe
    Input eval_cf_df: DataFrame with all CF
    Input eval_x_df: DataFrame containing all the instances of interest information
    Output full_df: DataFrame containing all the instances of interest and corresponding CFs information
    """
    full_df = pd.concat((eval_x_df, eval_cf_df),axis=1)
    full_df_cf_list = full_df['cf'].tolist()
    full_df_x_list = full_df['x'].tolist()
    full_df_method = full_df['cf_method']
    full_df_metrics = full_df[['proximity','sparsity','accuracy']]
    cf_df = pd.concat(full_df_cf_list)
    x_df = pd.concat(full_df_x_list)
    cf_df = pd.concat((full_df_method,cf_df,full_df_metrics),axis=1)
    return x_df, cf_df

def extract_values_labels(cf_df, method, feat, feat_val, metric, protected_feat):
    """
    Method that extracts the values of the metric and method of interest in the feature value specified
    """
    feat_val_name = protected_feat[feat][np.round(feat_val,2)]
    feat_metric_method_values = cf_df[(cf_df['cf_method'] == method) & (cf_df[feat] == feat_val)][metric]
    return feat_metric_method_values, method+':'+feat_val_name

def metric_differences_plot(datasets, methods_to_run, metrics):
    """
    Method that plots the metric differences among features, datasets and methods
    """
    for data_str in datasets:
        eval_obj = load_obj(data_str)
        eval_x_df = eval_obj.all_x_data
        eval_cf_df = eval_obj.all_cf_data
        protected_feat = eval_obj.feat_protected
        protected_feat_keys = list(protected_feat.keys())
        x_df, cf_df = extract_x_cd_df(eval_cf_df, eval_x_df)
        for feat in x_df.columns:
            feat_unique_val = x_df[feat].unique()
            for metric in metrics:
                metric_feat_list = []
                metric_feat_labels = []
                prot_feat_found = False
                if feat in protected_feat_keys:
                    feat_name = feat
                    prot_feat_found = True
                elif feat[:-4] in protected_feat_keys:
                    feat_name = feat[:-4]
                    prot_feat_found = True
                for method in methods_to_run:
                    for feat_val in feat_unique_val:
                        feat_metric_method_values, box_label = extract_values_labels(cf_df, method, feat_name, feat_val, metric, protected_feat)
                        metric_feat_list.append(feat_metric_method_values)
                        metric_feat_labels.append(box_label)
                if prot_feat_found:
                    plt.boxplot(metric_feat_list, notch=True, vert=False, labels = metric_feat_labels)
                    plt.savefig(results_cf_plots_dir+f'{data_str}_{feat}_{metric}_fairness_.png',dpi=400)

datasets = ['compass','credit','adult','german','heart']  # Name of the dataset to be analyzed ['synthetic_severe_disease','synthetic_athlete','ionosphere','compass','credit','adult','german','heart']
methods_to_run = ['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice'] #['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice']
metrics = ['proximity','sparsity','accuracy']

metric_differences_plot(datasets, methods_to_run, metrics)
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pickle
from support import path_here, results_cf_obj_dir, results_cf_plots_dir
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def load_obj(file_name):
    """
    Method to read an Evaluator object containing the evaluation results for all the instances of a given dataset
    """
    with open(results_cf_obj_dir+file_name+'_eval.pkl', 'rb') as input:
        evaluator_obj = pickle.load(input)
    return evaluator_obj

def extract_x_cd_df(eval_cf_df,eval_x_df, cf_metrics):
    """
    Method that joins accuracy, proximity and sparsity from the eval object and outputs the join as a single dataframe
    Input eval_cf_df: DataFrame with all CF
    Input eval_x_df: DataFrame containing all the instances of interest information
    Output full_df: DataFrame containing all the instances of interest and corresponding CFs information
    """
    eval_cf_df.dropna(inplace=True)
    full_df_cf_list = eval_cf_df['cf'].tolist()
    full_df_x_list = eval_x_df['x'].tolist()
    cf_df = pd.concat(full_df_cf_list)
    x_df = pd.concat(full_df_x_list)
    cf_df = cf_df.reset_index(drop=True)
    full_df_metrics = eval_cf_df[['cf_method']+cf_metrics]
    full_df_metrics = full_df_metrics.reset_index(drop=True)
    x_df = x_df.reset_index(drop=True)
    full_x_metrics = eval_x_df['accuracy']
    full_x_metrics = full_x_metrics.reset_index(drop=True)
    cf_df = pd.concat((cf_df,full_df_metrics),axis=1)
    x_df = pd.concat((x_df,full_x_metrics),axis=1)
    return x_df, cf_df

def extract_values_labels(cf_df, method, feat, df_feat_name, feat_val, metric, protected_feat):
    """
    Method that extracts the values of the metric and method of interest in the feature value specified
    """
    feat_val_name = protected_feat[feat][np.round(feat_val,2)]
    feat_metric_method_values = cf_df[(cf_df['cf_method'] == method) & (cf_df[df_feat_name] == feat_val)][metric]
    return feat_metric_method_values.values

def create_handles(feat, feat_unique_val, colors, protected_feat):
    """
    Method that creates legend handles to print in the image
    """
    list_handles = []
    for i in range(len(feat_unique_val)):
        handle = Line2D([0], [0], color=colors[i], lw=2, label=protected_feat[feat][np.round(feat_unique_val[i],2)])
        list_handles.extend([handle])
    return list_handles

def get_data_names(datasets):
    """
    Method that gets the names of the datasets for plotting
    """
    data_dict = {}
    for i in datasets:
        if i == 'compass':
            data_dict[i] = 'Compas'
        elif i == 'credit':
            data_dict[i] = 'Credit'
        elif i == 'adult':
            data_dict[i] = 'Adult'
        elif i == 'german':
            data_dict[i] = 'German'
        elif i == 'heart':
            data_dict[i] = 'Heart'
    return data_dict

def get_metric_names(metrics):
    """
    Method that gets the names of the metrics for plotting
    """
    metric_dict = {}
    for i in metrics:
        if i == 'proximity':
            metric_dict[i] = 'Proximity (Lower is better)'
        elif i == 'sparsity':
            metric_dict[i] = 'Sparsity (Higher is better)'
    return metric_dict

def get_methods_names(methods):
    """
    Method that gets the names of the CF methods for plotting
    """
    method_dict = {}
    for i in methods:
        if i == 'nn':
            method_dict[i] = 'NN'
        elif i == 'mo':
            method_dict[i] = 'MO'
        elif i == 'ft':
            method_dict[i] = 'FT'
        elif i == 'rt':
            method_dict[i] = 'RT'
        elif i == 'gs':
            method_dict[i] = 'GS'
        elif i == 'dice':
            method_dict[i] = 'DiCE'
        elif i == 'face':
            method_dict[i] = 'FACE'
        elif i == 'dice':
            method_dict[i] = 'DiCE'
        elif i == 'mace':
            method_dict[i] = 'MACE'
        elif i == 'cchvae':
            method_dict[i] = 'CCHVAE'
        elif i == 'jce_prox':
            method_dict[i] = 'JUICEP'
        elif i == 'jce_spar':
            method_dict[i] = 'JUICES'
    return method_dict

def metric_differences_plot(datasets, methods_to_run, cf_metrics, colors):
    """
    Method that plots the metric differences among features, datasets and methods
    """
    methods_names = get_methods_names(methods_to_run)
    dataset_names = get_data_names(datasets)
    metric_names = get_metric_names(cf_metrics)
    for data_str in datasets:
        eval_obj = load_obj(data_str)
        eval_x_df = eval_obj.all_x_data
        eval_cf_df = eval_obj.all_cf_data
        protected_feat = eval_obj.feat_protected
        protected_feat_keys = list(protected_feat.keys())
        x_df, cf_df = extract_x_cd_df(eval_cf_df, eval_x_df, cf_metrics)
        for feat in x_df.columns:
            feat_unique_val = x_df[feat].unique()
            xaxis_pos_labels = np.arange((len(feat_unique_val)-1)/2,len(methods_to_run)*len(feat_unique_val),len(feat_unique_val))
            xaxis_pos_bars = np.arange(len(methods_to_run)*len(feat_unique_val))
            for metric in cf_metrics:
                metric_feat_mean_list = []
                metric_feat_std_list = []
                metric_feat_labels = []
                colors_plot = []
                prot_feat_found = False
                if feat in protected_feat_keys:
                    feat_name = feat
                    prot_feat_found = True
                elif feat[:-4] in protected_feat_keys:
                    feat_name = feat[:-4]
                    prot_feat_found = True
                if prot_feat_found:
                    for method in methods_to_run:
                        for feat_idx in range(len(feat_unique_val)):
                            feat_metric_method_values = extract_values_labels(cf_df, method, feat_name, feat, feat_unique_val[feat_idx], metric, protected_feat)
                            metric_feat_mean_list.append(np.mean(feat_metric_method_values))
                            metric_feat_std_list.append(np.std(feat_metric_method_values,ddof=1))
                            colors_plot.append(colors[feat_idx])
                        metric_feat_labels.append(methods_names[method])
                    legend_elements = create_handles(feat_name, feat_unique_val, colors, protected_feat)
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.bar(xaxis_pos_bars, metric_feat_mean_list, yerr=metric_feat_std_list, color = colors_plot)
                    ax.set_xticks(xaxis_pos_labels, labels=metric_feat_labels)
                    ax.legend(handles=legend_elements)
                    ax.set_title(f'{dataset_names[data_str]} {metric_names[metric]} {feat_name}')
                    plt.savefig(results_cf_plots_dir+f'{data_str}_{feat_name}_{metric}_fairness.png',dpi=400)


datasets = ['compass']  # Name of the dataset to be analyzed ['compass','credit','adult','german','heart']
methods_to_run = ['nn','mo','ft','rt','gs','dice','cchvae','jce_prox','jce_spar'] #['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice']
colors = ['red', 'green', 'blue', 'pink', 'gold', 'cyan']
cf_metrics = ['proximity']

metric_differences_plot(datasets, methods_to_run, cf_metrics, colors)
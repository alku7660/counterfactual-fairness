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
    with open(results_cf_obj_dir+file_name+'_mutability_eval.pkl', 'rb') as input:
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
    full_df_metrics = eval_cf_df[['cf_method','instance_index','valid']+cf_metrics]
    full_df_metrics = full_df_metrics.reset_index(drop=True)
    x_df = x_df.reset_index(drop=True)
    full_x_metrics = eval_x_df[['accuracy','instance_index']]
    full_x_metrics = full_x_metrics.reset_index(drop=True)
    cf_df = pd.concat((cf_df,full_df_metrics),axis=1)
    x_df = pd.concat((x_df,full_x_metrics),axis=1)
    return x_df, cf_df

def extract_values_labels(cf_df, method, idx_list):
    """
    Method that extracts the values of the metric and method of interest in the feature value specified
    """
    feat_metric_method_data = cf_df[(cf_df['cf_method'] == method) & (cf_df['instance_index'].isin(idx_list))]
    return feat_metric_method_data

def extract_changed_values_ratio(x_df, cf_df, method, feat, idx_list):
    """
    Method that extracts the ratio of values changed for all sensitive features, for each method
    """
    cf_method_df = cf_df[cf_df['cf_method'] == method]
    total_instances = len(idx_list)
    counter_changed = 0
    for idx in idx_list:
        x_idx = x_df[x_df['instance_index'] == idx]
        cf_idx = cf_method_df[cf_method_df['instance_index'] == idx]
        if x_idx[feat].values != cf_idx[feat].values and cf_idx['valid'].values:
            counter_changed += 1
    return counter_changed / total_instances

def extract_number_idx_instances_feat_val(x_df, df_feat_name, feat_unique_val):
    """
    Method that extracts the number of instances per value of a feature of interest
    """
    len_feat_values, idx_feat_values = [], []
    for i in range(len(feat_unique_val)):
        feat_values = x_df[x_df[df_feat_name] == feat_unique_val[i]]
        feat_values_idx = feat_values['instance_index'].values
        len_feat_values.append(len(feat_values))
        idx_feat_values.append(feat_values_idx)
    return len_feat_values, idx_feat_values

def create_handles(feat, feat_unique_val, colors, protected_feat, len_feat_values):
    """
    Method that creates legend handles to print in the image
    """
    list_handles = []
    total_instances = np.sum(len_feat_values)
    for i in range(len(feat_unique_val)):
        handle = Line2D([0], [0], color=colors[i], lw=2, label=f'{protected_feat[feat][np.round(feat_unique_val[i],2)]} ({len_feat_values[i]} examples, {np.round(len_feat_values[i]*100/total_instances,1)}%)')
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
            metric_dict[i] = 'Burden (Lower is better)'
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
        elif i == 'mutable-nn':
            method_dict[i] = 'Mutable NN'
        elif i == 'mo':
            method_dict[i] = 'MO'
        elif i == 'mutable-mo':
            method_dict[i] = 'Mutable MO'
        elif i == 'ft':
            method_dict[i] = 'FT'
        elif i == 'rt':
            method_dict[i] = 'RT'
        elif i == 'mutable-rt':
            method_dict[i] = 'Mutable RT'
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
        elif i == 'mutable-jce_prox':
            method_dict[i] = 'Mutable JUICEP'
        elif i == 'jce_spar':
            method_dict[i] = 'JUICES'
        elif i == 'mutable-jce_spar':
            method_dict[i] = 'Mutable JUICES'
    return method_dict

def get_feature_name(feat, protected_feat_keys):
    """
    Method to obtain the feature name in the original dataset
    """
    feat_name, prot_feat_found = None, False
    if feat in protected_feat_keys:
        feat_name = feat
        prot_feat_found = True
    elif feat[:-4] in protected_feat_keys:
        feat_name = feat[:-4]
        prot_feat_found = True
    return feat_name, prot_feat_found

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
            feat_name, prot_feat_found = get_feature_name(feat, protected_feat_keys)
            if feat_name is None:
                continue
            feat_unique_val = x_df[feat].unique()
            len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(x_df, feat, feat_unique_val)
            xaxis_pos_labels = np.arange((len(feat_unique_val)-1)/2,len(methods_to_run)*len(feat_unique_val),len(feat_unique_val))
            xaxis_pos_box = np.arange(len(methods_to_run)*len(feat_unique_val))
            for metric in cf_metrics:
                method_feat_values_list = []
                method_feat_ratio_list = []
                method_feat_labels = []
                method_feat_valid_list = []
                colors_plot = []
                if prot_feat_found:
                    for method in methods_to_run:
                        for feat_idx in range(len(feat_unique_val)):
                            box_feat_val_pos = xaxis_pos_box[feat_idx::len(feat_unique_val)]
                            feat_method_data = cf_df[(cf_df['cf_method'] == method) & (cf_df['instance_index'].isin(idx_feat_values[feat_idx]))]
                            feat_method_data_values = feat_method_data[metric].values
                            feat_method_data_valid = np.sum(feat_method_data['valid'].values)*100/len(feat_method_data)
                            feat_ratio_values_changed = extract_changed_values_ratio(x_df, cf_df, method, feat, idx_feat_values[feat_idx])
                            method_feat_ratio_list.append(feat_ratio_values_changed)
                            method_feat_values_list.append(feat_method_data_values)
                            method_feat_valid_list.append(feat_method_data_valid)
                            colors_plot.append(colors[feat_idx])
                            ax.boxplot(positions=box_feat_val_pos, x=method_feat_values_list)
                        method_feat_labels.append(methods_names[method])
                    legend_elements = create_handles(feat_name, feat_unique_val, colors, protected_feat, len_feat_values)
                    fig, ax = plt.subplots(figsize=(8,6))
                    # counter = 0
                    # for pl in ax_plot.get_xticks():
                    #     ax.text(x=pl, y=0.1*min([min(i) for i in method_feat_values_list]), s='{}'.format(np.round(method_feat_ratio_list[counter]*100,1)), ha='center')
                    #     counter += 1
                    ax.set_xticks(xaxis_pos_labels, labels=method_feat_labels)
                    ax.set_xticklabels(method_feat_labels, rotation = 10, ha='center')
                    # box = ax.get_position()
                    # ax.set_position([box.x0, box.y0, box.width, box.height*0.5])
                    ax.set_title(f'{dataset_names[data_str]} Dataset: {metric_names[metric]} by {feat_name}')
                    ax.set_ylabel('CF Distance to Instance of Interest (Euclidean)')
                    ax.set_xlabel('Counterfactual Method')
                    axvalid = ax.twinx()
                    color_secax = 'tab:blue'
                    axvalid.set_ylabel('Percentage of attainable CFs (%)',color=color_secax)
                    axvalid.plot(xaxis_pos_bars, method_feat_valid_list, color=color_secax, linestyle='', marker='D')
                    axvalid.tick_params(axis='y', labelcolor=color_secax)
                    ax.legend(handles=legend_elements, loc=(-0.1,-0.1*len(legend_elements)))
                    plt.tight_layout()
                    plt.savefig(results_cf_plots_dir+f'{data_str}_{feat_name}_{metric}_fairness.png',dpi=400)

def accuracy_differences_plot(datasets, methods_to_run, cf_metrics, colors):
    """
    Method that plots the accuracy differences among features, datasets and methods
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
                    ax.set_title(f'{dataset_names[data_str]}: {metric_names[metric]} by {feat_name}')
                    ax.set_ylabel('CF Distance to Instance of Interest (Euclidean)')
                    ax.set_xlabel('Counterfactual Generation Method')
                    plt.savefig(results_cf_plots_dir+f'{data_str}_{feat_name}_{metric}_fairness.png',dpi=400)

datasets = ['adult','compass']  # Name of the dataset to be analyzed ['compass','credit','adult','german','heart'] ,'jce_prox','mutable_jce_prox'
methods_to_run = ['nn','mutable-nn','mo','mutable-mo','rt','mutable-rt'] #['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice']
colors = ['red', 'green', 'blue', 'pink', 'gold', 'cyan']
cf_metrics = ['proximity']

metric_differences_plot(datasets, methods_to_run, cf_metrics, colors)
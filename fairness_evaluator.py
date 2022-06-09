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
from matplotlib import cm

def load_obj(file_name, study_str, file):
    """
    Method to read an Evaluator object containing the evaluation results for all the instances of a given dataset
    """
    with open(results_cf_obj_dir+file_name+'_'+study_str+'_'+file+'.pkl', 'rb') as input:
        evaluator_obj = pickle.load(input)
    return evaluator_obj

def extract_x_cd_df(eval_cf_df, eval_x_df, cf_metrics=None, data_obj=None):
    """
    Method that joins accuracy, proximity and sparsity from the eval object and outputs the join as a single dataframe
    Input eval_cf_df: DataFrame with all CF
    Input eval_x_df: DataFrame containing all the instances of interest information
    Output full_df: DataFrame containing all the instances of interest and corresponding CFs information
    """
    eval_cf_df.dropna(inplace=True)
    full_df_cf_list = eval_cf_df['cf'].tolist()
    full_df_original_cf_list = eval_cf_df['original_cf'].tolist()
    full_df_x_list = eval_x_df['x'].tolist()
    cf_df = pd.concat(full_df_cf_list)
    original_cf_df = pd.concat(full_df_original_cf_list)
    x_df = pd.concat(full_df_x_list)
    if data_obj is None:
        full_df_x_original_list = eval_x_df['original_x'].tolist()
        original_x_df = pd.concat(full_df_x_original_list)
    else:
        original_x_df = data_obj.test_pd.loc[x_df.index]
    if cf_metrics is not None:
        full_df_metrics = eval_cf_df[['cf_method','valid']+cf_metrics]
    else:
        full_df_metrics = eval_cf_df[['cf_method','valid']]
    full_df_metrics.index = cf_df.index
    full_x_metrics = eval_x_df[['accuracy']]
    full_x_metrics.index = x_df.index
    cf_df = pd.concat((cf_df,full_df_metrics),axis=1)
    x_df = pd.concat((x_df,full_x_metrics),axis=1)
    original_cf_df = pd.concat((original_cf_df,full_df_metrics),axis=1)
    original_x_df = pd.concat((original_x_df,full_x_metrics),axis=1)
    return x_df, cf_df, original_x_df, original_cf_df

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
        x_idx = x_df[x_df.index == idx]
        cf_idx = cf_method_df[cf_method_df.index == idx]
        if x_idx[feat].values != cf_idx[feat].values and cf_idx['valid'].values:
            counter_changed += 1
    return counter_changed*100 / total_instances

def extract_number_idx_instances_feat_val(original_x_df, feat_name, feat_unique_val):
    """
    Method that extracts the number of instances per value of a feature of interest
    """
    len_feat_values, idx_feat_values = [], []
    for i in range(len(feat_unique_val)):
        feat_values = original_x_df[original_x_df[feat_name] == feat_unique_val[i]]
        feat_values_idx = feat_values.index.tolist()
        len_feat_values.append(len(feat_values))
        idx_feat_values.append(feat_values_idx)
    return len_feat_values, idx_feat_values

def extract_accuracy_feat_val(x_df, idx_list):
    """
    Method that extracts the accuracy of the instances per protected feature group
    """
    x_feat_df = x_df[x_df.index.isin(idx_list)]
    x_feat_df_acc = x_feat_df['accuracy']
    sum_x_feat_df_acc = np.sum(x_feat_df_acc)
    return sum_x_feat_df_acc/len(idx_list)

def create_box_bar_plot_handles(feat, feat_unique_val, colors, protected_feat, len_feat_values):
    """
    Method that creates legend handles to print in the image
    """
    list_handles = []
    total_instances = np.sum(len_feat_values)
    for i in range(len(feat_unique_val)):
        handle = Line2D([0], [0], color=colors[i], lw=2, label=f'{protected_feat[feat][np.round(feat_unique_val[i],2)]} ({len_feat_values[i]} examples, {np.round(len_feat_values[i]*100/total_instances,1)}%)')
        list_handles.extend([handle])
    return list_handles

def create_metric_burden_handles(protected_feat_keys, colors):
    """
    Method that creates legend handles to print in the image
    """
    list_handles = []
    for i in range(len(protected_feat_keys)):
        feat = protected_feat_keys[i]
        handle = Line2D([0], [0], color=colors[i], lw=2, label=f'{feat}')
        list_handles.extend([handle])
    return list_handles

def get_data_names(datasets):
    """
    Method that gets the names of the datasets for plotting
    """
    'adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law'
    data_dict = {}
    for i in datasets:
        if i == 'adult':
            data_dict[i] = 'Adult'
        elif i == 'kdd_census':
            data_dict[i] = 'Census'
        elif i == 'german':
            data_dict[i] = 'German'
        elif i == 'dutch':
            data_dict[i] = 'Dutch'
        elif i == 'bank':
            data_dict[i] = 'Bank'
        elif i == 'credit':
            data_dict[i] = 'Credit'
        elif i == 'compass':
            data_dict[i] = 'Compas'
        elif i == 'diabetes':
            data_dict[i] = 'Diabetes'
        elif i == 'student':
            data_dict[i] = 'Student'
        elif i == 'oulad':
            data_dict[i] = 'Oulad'
        elif i == 'law':
            data_dict[i] = 'Law'
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
        elif i == 'juice':
            method_dict[i] = 'JUICEP'
        elif i == 'mutable-juice':
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

def method_box_plot(datasets, methods_to_run, metric, colors):
    """
    Method that plots the metric differences among features, datasets and methods
    """
    methods_names = get_methods_names(methods_to_run)
    dataset_names = get_data_names(datasets)
    metric_names = get_metric_names([metric])
    for data_str in datasets:
        eval_obj = load_obj(data_str, 'mutability', 'eval')
        eval_x_df = eval_obj.all_x_data
        eval_cf_df = eval_obj.all_cf_data
        protected_feat = eval_obj.feat_protected
        protected_feat_keys = list(protected_feat.keys())
        x_df, cf_df, original_x_df, original_cf_df = extract_x_cd_df(eval_cf_df, eval_x_df, [metric])
        for feat in protected_feat_keys:
            feat_unique_val = original_x_df[feat].unique()
            len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
            fig, ax = plt.subplots(figsize=(8,6))
            xaxis_pos_labels = np.arange((len(feat_unique_val)-1)/2,len(methods_to_run)*len(feat_unique_val),len(feat_unique_val))
            xaxis_pos_box = np.arange(len(methods_to_run)*len(feat_unique_val))
            method_feat_labels = []
            for method_idx in range(len(methods_to_run)):
                for feat_idx in range(len(feat_unique_val)):
                    box_feat_val_pos = xaxis_pos_box[method_idx*len(feat_unique_val)+feat_idx]
                    feat_method_data = cf_df[(cf_df['cf_method'] == methods_to_run[method_idx]) & (cf_df.index.isin(idx_feat_values[feat_idx]))]
                    feat_method_data_values = feat_method_data[metric].values
                    c = colors[feat_idx]
                    ax.boxplot(x=feat_method_data_values, positions=[box_feat_val_pos], boxprops=dict(color=c),
                               capprops=dict(color=c), showfliers=False, whiskerprops=dict(color=c),
                               medianprops=dict(color=c), widths=0.9, showmeans=True,
                               meanprops=dict(markerfacecolor=c, markeredgecolor=c, marker='D'), flierprops=dict(markeredgecolor=c), notch=False)
                method_feat_labels.append(methods_names[methods_to_run[method_idx]])
            legend_elements = create_box_bar_plot_handles(feat, feat_unique_val, colors, protected_feat, len_feat_values)
            ax.set_xticks(xaxis_pos_labels, labels=method_feat_labels)
            ax.set_xticklabels(method_feat_labels, rotation = 10, ha='center')
            ax.set_title(f'{dataset_names[data_str]} Dataset: {metric_names[metric]} by {feat}')
            ax.set_ylabel('Burden (Lower is Better)')
            ax.set_xlabel('Counterfactual Method')
            ax.legend(handles=legend_elements) #loc=(-0.1,-0.1*len(legend_elements))
            plt.tight_layout()
            plt.savefig(results_cf_plots_dir+f'{data_str}_{feat}_{metric}_method_feat_burden.png',dpi=400)

def attainable_cf_plot(datasets, methods_to_run):
    """
    Method that plots the percentage of attainable CFs given feasibility constraints and whther or not to consider feature mutability
    """
    methods_names = get_methods_names(methods_to_run)
    dataset_names = get_data_names(datasets)
    for data_str in datasets:
        eval_obj = load_obj(data_str, 'mutability', 'eval')
        eval_x_df = eval_obj.all_x_data
        eval_cf_df = eval_obj.all_cf_data
        protected_feat = eval_obj.feat_protected
        protected_feat_keys = list(protected_feat.keys())
        x_df, cf_df, original_x_df, original_cf_df = extract_x_cd_df(eval_cf_df, eval_x_df)
        for feat in protected_feat_keys:
            feat_unique_val = original_x_df[feat].unique()
            len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
            fig, ax = plt.subplots(figsize=(8,6))
            xaxis_pos_labels = np.arange((len(feat_unique_val)-1)/2,len(methods_to_run)*len(feat_unique_val),len(feat_unique_val))
            xaxis_pos_bar = np.arange(len(methods_to_run)*len(feat_unique_val))
            method_feat_labels = []
            method_feat_valid_list = []
            color_list = []
            for method_idx in range(len(methods_to_run)):
                for feat_idx in range(len(feat_unique_val)):
                    feat_method_data = cf_df[(cf_df['cf_method'] == methods_to_run[method_idx]) & (cf_df.index.isin(idx_feat_values[feat_idx]))]
                    feat_method_data_valid = np.sum(feat_method_data['valid'].values)*100/len(feat_method_data)
                    method_feat_valid_list.append(feat_method_data_valid)
                    color_list.append(colors[feat_idx])
                method_feat_labels.append(methods_names[methods_to_run[method_idx]])
            legend_elements = create_box_bar_plot_handles(feat, feat_unique_val, colors, protected_feat, len_feat_values)
            ax.bar(x=xaxis_pos_bar, height=method_feat_valid_list, color=color_list)
            ax.set_xticks(xaxis_pos_labels, labels=method_feat_labels)
            ax.set_xticklabels(method_feat_labels, rotation = 10, ha='center')
            ax.set_title(f'{dataset_names[data_str]} Dataset: Attainable CFs by {feat}')
            ax.set_ylabel('Attainable CFs (%)')
            ax.set_xlabel('Counterfactual Method')
            ax.legend(handles=legend_elements) #loc=(-0.1,-0.1*len(legend_elements))
            plt.tight_layout()
            plt.savefig(results_cf_plots_dir+f'{data_str}_{feat}_attainable_change.png',dpi=400)

def feature_ratio_change_cf_plot(datasets, methods_to_run):
    """
    Method that plots the percentage of attainable CFs given feasibility constraints and whther or not to consider feature mutability
    """
    methods_names = get_methods_names(methods_to_run)
    dataset_names = get_data_names(datasets)
    for data_str in datasets:
        eval_obj = load_obj(data_str, 'mutability', 'eval')
        eval_x_df = eval_obj.all_x_data
        eval_cf_df = eval_obj.all_cf_data
        protected_feat = eval_obj.feat_protected
        protected_feat_keys = list(protected_feat.keys())
        x_df, cf_df, original_x_df, original_cf_df = extract_x_cd_df(eval_cf_df, eval_x_df)
        for feat in protected_feat_keys:
            feat_unique_val = original_x_df[feat].unique()
            len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
            fig, ax = plt.subplots(figsize=(8,6))
            xaxis_pos_labels = np.arange((len(feat_unique_val)-1)/2,len(methods_to_run)*len(feat_unique_val),len(feat_unique_val))
            xaxis_pos_bar = np.arange(len(methods_to_run)*len(feat_unique_val))
            method_feat_labels = []
            method_feat_changed_ratio_list = []
            color_list = []
            for method_idx in range(len(methods_to_run)):
                for feat_idx in range(len(feat_unique_val)):
                    feat_ratio_values_changed = extract_changed_values_ratio(original_x_df, original_cf_df, methods_to_run[method_idx], feat, idx_feat_values[feat_idx])
                    method_feat_changed_ratio_list.append(feat_ratio_values_changed)
                    color_list.append(colors[feat_idx])
                method_feat_labels.append(methods_names[methods_to_run[method_idx]])
            legend_elements = create_box_bar_plot_handles(feat, feat_unique_val, colors, protected_feat, len_feat_values)
            ax.bar(x=xaxis_pos_bar, height=method_feat_changed_ratio_list, color=color_list)
            ax.set_xticks(xaxis_pos_labels, labels=method_feat_labels)
            ax.set_xticklabels(method_feat_labels, rotation = 10, ha='center')
            ax.set_title(f'{dataset_names[data_str]} Dataset: Changed Values by {feat}')
            ax.set_ylabel('Instances with sensitive feature changed (%)')
            ax.set_xlabel('Counterfactual Method')
            ax.legend(handles=legend_elements) #loc=(-0.1,-0.1*len(legend_elements))
            plt.tight_layout()
            plt.savefig(results_cf_plots_dir+f'{data_str}_{feat}_feat_change.png',dpi=400)

def accuracy_burden_plot(datasets, method, metric, colors):
    """
    Method that plots the accuracy differences among features, datasets and methods
    """
    methods_names = get_methods_names([method])
    dataset_names = get_data_names(datasets)
    for data_str in datasets:
        eval_obj = load_obj(data_str, 'mutability', 'eval')
        eval_x_df = eval_obj.all_x_data
        eval_cf_df = eval_obj.all_cf_data
        protected_feat = eval_obj.feat_protected
        protected_feat_keys = list(protected_feat.keys())
        x_df, cf_df, original_x_df, original_cf_df = extract_x_cd_df(eval_cf_df, eval_x_df, [metric])
        fig, ax = plt.subplots(figsize=(8,6))
        method_feat_valid_list = []
        for prot_feat_idx in range(len(protected_feat_keys)):
            feat = protected_feat_keys[prot_feat_idx]
            feat_unique_val = original_x_df[feat].unique()
            len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
            pos_list = []
            mean_data_val_list = []
            for feat_idx in range(len(feat_unique_val)):
                feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_idx],2)]
                feat_method_data = cf_df[(cf_df['cf_method'] == method) & (cf_df.index.isin(idx_feat_values[feat_idx]))]
                feat_method_data_values = feat_method_data[metric].values
                accuracy_feat_val = np.round(extract_accuracy_feat_val(x_df, idx_feat_values[feat_idx]),2)
                pos_list.append(accuracy_feat_val)
                mean_data_val_list.append(np.mean(feat_method_data_values))
                feat_method_data_valid = np.sum(feat_method_data['valid'].values)*100/len(feat_method_data)
                method_feat_valid_list.append(feat_method_data_valid)
                c = colors[prot_feat_idx]
                ax.text(x=accuracy_feat_val, y=np.mean(feat_method_data_values), bbox=dict(ec=c,fc='none'),
                        s=f'{feat_val_name}', fontstyle='italic', color=c, size=9)
            ax.scatter(x=pos_list, y=mean_data_val_list, color=colors[prot_feat_idx], s=25)
        legend_handles = create_metric_burden_handles(protected_feat_keys, colors)
        y_min, y_max = ax.get_ylim()
        # ax.set_ylim(y_max*(1.01),y_min*(0.99))
        ax.set_ylim(y_min*(0.99),y_max*(1.01))
        ax.set_title(f'{dataset_names[data_str]} Dataset: {methods_names[method]} Method')
        ax.set_ylabel('Burden (Lower is Better)')
        ax.set_xlabel('Classification Accuracy')
        ax.legend(handles=legend_handles) #loc=(-0.1,-0.1*len(legend_elements))
        plt.tight_layout()
        plt.savefig(results_cf_plots_dir+f'{data_str}_accuracy_burden_fairness.png',dpi=400)

def statistical_parity_burden_plot(datasets, method, metric, colors):
    """
    Method that plots the accuracy differences among features, datasets and methods
    """
    methods_names = get_methods_names([method])
    dataset_names = get_data_names(datasets)
    for data_str in datasets:
        eval_obj = load_obj(data_str, 'mutability', 'eval')
        eval_x_df = eval_obj.all_x_data
        eval_cf_df = eval_obj.all_cf_data
        protected_feat = eval_obj.feat_protected
        protected_feat_keys = list(protected_feat.keys())
        x_df, cf_df, original_x_df, original_cf_df = extract_x_cd_df(eval_cf_df, eval_x_df, [metric])
        fig, ax = plt.subplots(figsize=(8,6))
        method_feat_valid_list = []
        for prot_feat_idx in range(len(protected_feat_keys)):
            feat = protected_feat_keys[prot_feat_idx]
            feat_unique_val = original_x_df[feat].unique()
            len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
            pos_list = []
            mean_data_val_list = []
            for feat_idx in range(len(feat_unique_val)):
                feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_idx],2)]
                feat_method_data = cf_df[(cf_df['cf_method'] == method) & (cf_df.index.isin(idx_feat_values[feat_idx]))]
                feat_method_data_values = feat_method_data[metric].values
                stat_parity_feat_val = eval_obj.stat_parity[feat][feat_val_name]
                pos_list.append(stat_parity_feat_val)
                mean_data_val_list.append(np.mean(feat_method_data_values))
                feat_method_data_valid = np.sum(feat_method_data['valid'].values)*100/len(feat_method_data)
                method_feat_valid_list.append(feat_method_data_valid)
                c = colors[prot_feat_idx]
                ax.text(x=stat_parity_feat_val, y=np.mean(feat_method_data_values), bbox=dict(ec=c,fc='none'),
                        s=feat_val_name, fontstyle='italic', color=c, size=9)
            ax.scatter(x=pos_list, y=mean_data_val_list, color=colors[prot_feat_idx], s=25)
        legend_handles = create_metric_burden_handles(protected_feat_keys, colors)
        y_min, y_max = ax.get_ylim()
        # ax.set_ylim(y_max*(1.01),y_min*(0.99))
        ax.set_ylim(y_min*(0.99),y_max*(1.01))
        ax.set_title(f'{dataset_names[data_str]} Dataset: {methods_names[method]} Method')
        ax.set_ylabel('Burden (Lower is Better)')
        ax.set_xlabel('Statistical Parity')
        ax.legend(handles=legend_handles) #loc=(-0.1,-0.1*len(legend_elements))
        plt.tight_layout()
        plt.savefig(results_cf_plots_dir+f'{data_str}_stat_parity_burden_fairness.png',dpi=400)

def equalized_odds_burden_plot(datasets, method, metric, colors):
    """
    Method that plots the accuracy differences among features, datasets and methods
    """
    methods_names = get_methods_names([method])
    dataset_names = get_data_names(datasets)
    for data_str in datasets:
        eval_obj = load_obj(data_str, 'mutability', 'eval')
        eval_x_df = eval_obj.all_x_data
        eval_cf_df = eval_obj.all_cf_data
        protected_feat = eval_obj.feat_protected
        protected_feat_keys = list(protected_feat.keys())
        x_df, cf_df, original_x_df, original_cf_df = extract_x_cd_df(eval_cf_df, eval_x_df, [metric])
        fig, ax = plt.subplots(figsize=(8,6))
        method_feat_valid_list = []
        for prot_feat_idx in range(len(protected_feat_keys)):
            feat = protected_feat_keys[prot_feat_idx]
            feat_unique_val = original_x_df[feat].unique()
            len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
            pos_list = []
            mean_data_val_list = []
            for feat_idx in range(len(feat_unique_val)):
                feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_idx],2)]
                feat_method_data = cf_df[(cf_df['cf_method'] == method) & (cf_df.index.isin(idx_feat_values[feat_idx]))]
                feat_method_data_values = feat_method_data[metric].values
                stat_parity_feat_val = eval_obj.eq_odds[feat][feat_val_name]
                pos_list.append(stat_parity_feat_val)
                mean_data_val_list.append(np.mean(feat_method_data_values))
                feat_method_data_valid = np.sum(feat_method_data['valid'].values)*100/len(feat_method_data)
                method_feat_valid_list.append(feat_method_data_valid)
                c = colors[prot_feat_idx]
                ax.text(x=stat_parity_feat_val, y=np.mean(feat_method_data_values), bbox=dict(ec=c,fc='none'),
                        s=feat_val_name, fontstyle='italic', color=c, size=9)
            ax.scatter(x=pos_list, y=mean_data_val_list, color=colors[prot_feat_idx], s=25)
        legend_handles = create_metric_burden_handles(protected_feat_keys, colors)
        y_min, y_max = ax.get_ylim()
        # ax.set_ylim(y_max*(1.01),y_min*(0.99))
        ax.set_ylim(y_min*(0.99),y_max*(1.01))
        ax.set_title(f'{dataset_names[data_str]} Dataset: {methods_names[method]} Method')
        ax.set_ylabel('Burden (Lower is Better)')
        ax.set_xlabel('Equalized Odds')
        ax.legend(handles=legend_handles) #loc=(-0.1,-0.1*len(legend_elements))
        plt.tight_layout()
        plt.savefig(results_cf_plots_dir+f'{data_str}_eq_odds_burden_fairness.png',dpi=400)

def false_negative_plot(datasets, method, metric, colors):
    """
    Method that obtains false negative rate plots for each sensitive feature
    """
    methods_names = get_methods_names([method])
    dataset_names = get_data_names(datasets)
    for data_str in datasets:
        eval_obj = load_obj(data_str, 'fnr', 'eval')
        data_obj = load_obj(data_str, 'fnr', 'data')
        model_obj = load_obj(data_str, 'fnr', 'model')
        eval_x_df = eval_obj.all_x_data
        eval_cf_df = eval_obj.all_cf_data
        protected_feat = eval_obj.feat_protected
        protected_feat_keys = list(protected_feat.keys())
        x_df, cf_df, original_x_df, original_cf_df = extract_x_cd_df(eval_cf_df, eval_x_df, [metric], data_obj)
        desired_ground_truth_jce_test_pd = data_obj.jce_test_pd.loc[data_obj.test_target != data_obj.undesired_class]
        desired_ground_truth_test_pd = data_obj.test_pd.loc[data_obj.test_target != data_obj.undesired_class]
        desired_ground_truth_target = data_obj.test_target[data_obj.test_target != data_obj.undesired_class]
        predicted_label_desired_ground_truth_jce_test_pd = model_obj.jce_sel.predict(desired_ground_truth_jce_test_pd)
        false_undesired_jce_test_pd = desired_ground_truth_jce_test_pd.loc[predicted_label_desired_ground_truth_jce_test_pd == data_obj.undesired_class]
        false_undesired_test_pd = desired_ground_truth_test_pd.loc[predicted_label_desired_ground_truth_jce_test_pd == data_obj.undesired_class]
        false_undesired_target = desired_ground_truth_target[predicted_label_desired_ground_truth_jce_test_pd == data_obj.undesired_class]
        fig, ax = plt.subplots(figsize=(8,6))
        for prot_feat_idx in range(len(protected_feat_keys)):
            feat = protected_feat_keys[prot_feat_idx]
            feat_unique_val = desired_ground_truth_test_pd[feat].unique()
            len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
            x_pos_list = []
            mean_data_val_list = []
            # std_data_val_list = []
            for feat_idx in range(len(feat_unique_val)):
                feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_idx],2)]
                total_ground_truth_feat_val = np.sum(desired_ground_truth_test_pd[feat] == feat_unique_val[feat_idx])
                total_false_undesired_feat_val = np.sum(false_undesired_test_pd[feat] == feat_unique_val[feat_idx])
                fnr_feat_val = total_false_undesired_feat_val/total_ground_truth_feat_val
                x_pos_list.append(fnr_feat_val)
                feat_method_data = cf_df[(cf_df['cf_method'] == method) & (cf_df.index.isin(idx_feat_values[feat_idx]))]
                feat_method_data_values = feat_method_data[metric].values
                mean_data_val_list.append(np.mean(feat_method_data_values))
                # std_data_val_list.append(np.std(feat_method_data_values,ddof=1))
                c = colors[prot_feat_idx]
                ax.text(x=fnr_feat_val, y=np.mean(feat_method_data_values), bbox=dict(ec=c,fc='none'),
                        s=feat_val_name, fontstyle='italic', color=c, size=9)
            ax.scatter(x=x_pos_list, y=mean_data_val_list, color=colors[prot_feat_idx], s=25)
        legend_handles = create_metric_burden_handles(protected_feat_keys, colors)
        y_min, y_max = ax.get_ylim()
        # ax.set_ylim(y_max*(1.01),y_min*(0.99))
        ax.set_ylim(y_min*(0.99),y_max*(1.01))
        ax.set_title(f'{dataset_names[data_str]} Dataset: {methods_names[method]} Method')
        ax.set_ylabel('Burden (Lower is Better)')
        ax.set_xlabel('False Negative Ratio')
        ax.legend(handles=legend_handles) #loc=(-0.1,-0.1*len(legend_elements))
        plt.tight_layout()
        plt.savefig(results_cf_plots_dir+f'{data_str}_fnr_burden_fairness.png',dpi=400)

def accuracy_weighted_burden_plot(datasets, method, metric, colors):
    """
    Method that obtains the accuracy weighted burden for each method and each dataset
    """
    methods_names = get_methods_names([method])
    dataset_names = get_data_names(datasets)
    for data_str in datasets:
        eval_obj = load_obj(data_str, 'fnr', 'eval')
        data_obj = load_obj(data_str, 'fnr', 'data')
        model_obj = load_obj(data_str, 'fnr', 'model')
        eval_x_df = eval_obj.all_x_data
        eval_cf_df = eval_obj.all_cf_data
        protected_feat = eval_obj.feat_protected
        protected_feat_keys = list(protected_feat.keys())
        x_df, cf_df, original_x_df, original_cf_df = extract_x_cd_df(eval_cf_df, eval_x_df, [metric], data_obj)
        desired_ground_truth_jce_test_pd = data_obj.jce_test_pd.loc[data_obj.test_target != data_obj.undesired_class]
        desired_ground_truth_test_pd = data_obj.test_pd.loc[data_obj.test_target != data_obj.undesired_class]
        desired_ground_truth_target = data_obj.test_target[data_obj.test_target != data_obj.undesired_class]
        predicted_label_desired_ground_truth_jce_test_pd = model_obj.jce_sel.predict(desired_ground_truth_jce_test_pd)
        false_undesired_jce_test_pd = desired_ground_truth_jce_test_pd.loc[predicted_label_desired_ground_truth_jce_test_pd == data_obj.undesired_class]
        false_undesired_test_pd = desired_ground_truth_test_pd.loc[predicted_label_desired_ground_truth_jce_test_pd == data_obj.undesired_class]
        false_undesired_target = desired_ground_truth_target[predicted_label_desired_ground_truth_jce_test_pd == data_obj.undesired_class]
        fig, ax = plt.subplots(figsize=(8,6))
        for prot_feat_idx in range(len(protected_feat_keys)):
            feat = protected_feat_keys[prot_feat_idx]
            feat_unique_val = desired_ground_truth_test_pd[feat].unique()
            len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
            x_pos_list = []
            mean_data_val_list = []
            # std_data_val_list = []
            for feat_idx in range(len(feat_unique_val)):
                feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_idx],2)]
                total_ground_truth_feat_val = np.sum(desired_ground_truth_test_pd[feat] == feat_unique_val[feat_idx])
                total_false_undesired_feat_val = np.sum(false_undesired_test_pd[feat] == feat_unique_val[feat_idx])
                fnr_feat_val = total_false_undesired_feat_val/total_ground_truth_feat_val
                x_pos_list.append(fnr_feat_val)
                feat_method_data = cf_df[(cf_df['cf_method'] == method) & (cf_df.index.isin(idx_feat_values[feat_idx]))]
                feat_method_data_values = feat_method_data[metric].values
                mean_data_val_list.append(np.mean(feat_method_data_values))
                # std_data_val_list.append(np.std(feat_method_data_values,ddof=1))
                c = colors[prot_feat_idx]
                ax.text(x=fnr_feat_val, y=np.mean(feat_method_data_values), bbox=dict(ec=c,fc='none'),
                        s=feat_val_name, fontstyle='italic', color=c, size=9)
            ax.scatter(x=x_pos_list, y=mean_data_val_list, color=colors[prot_feat_idx], s=25)
        legend_handles = create_metric_burden_handles(protected_feat_keys, colors)
        y_min, y_max = ax.get_ylim()
        # ax.set_ylim(y_max*(1.01),y_min*(0.99))
        ax.set_ylim(y_min*(0.99),y_max*(1.01))
        ax.set_title(f'{dataset_names[data_str]} Dataset: {methods_names[method]} Method')
        ax.set_ylabel('Burden (Lower is Better)')
        ax.set_xlabel('False Negative Ratio')
        ax.legend(handles=legend_handles) #loc=(-0.1,-0.1*len(legend_elements))
        plt.tight_layout()
        plt.savefig(results_cf_plots_dir+f'{data_str}_fnr_burden_fairness.png',dpi=400)

datasets = ['adult','kdd_census','german','dutch','bank','credit','compass','diabetes','student','oulad','law']  # Name of the dataset to be analyzed ['adult','kdd_census','dutch','bank','compass']
methods_to_run = ['mutable-nn','mutable-mo','mutable-rt','cchvae'] #['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice']
colors = ['red', 'purple', 'tab:brown', 'blue', 'lightgreen', 'gold', 'orange']

attainable_cf_plot(datasets, methods_to_run)
feature_ratio_change_cf_plot(datasets, methods_to_run)
method_box_plot(datasets, methods_to_run, 'proximity', colors)
accuracy_burden_plot(datasets, 'mo', 'proximity', colors)
statistical_parity_burden_plot(datasets, 'mo', 'proximity', colors)
equalized_odds_burden_plot(datasets, 'mo', 'proximity', colors)
false_negative_plot(datasets, 'mo', 'proximity', colors)

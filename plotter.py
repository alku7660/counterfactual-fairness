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
from matplotlib.lines import Line2D
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter
matplotlib.rc('ytick', labelsize=9)
matplotlib.rc('xtick', labelsize=9)

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
            data_dict[i] = 'KDD Census'
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

def fnr_burden_plot(datasets, methods, metric, colors):
    """
    Method that obtains false negative rate plots for each sensitive feature
    """
    methods_names = get_methods_names(methods)
    dataset_names = get_data_names(datasets)
    fig, ax = plt.subplots(nrows=len(datasets),ncols=len(methods),sharex=False,sharey=False,figsize=(8,4.5))
    fig.supxlabel('$FNR_s$')
    fig.supylabel('$Burden_s$ (Lower is Better)')
    for i in range(len(datasets)):
        data_str = datasets[i]
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
        predicted_label_desired_ground_truth_jce_test_pd = model_obj.jce_sel.predict(desired_ground_truth_jce_test_pd)
        false_undesired_test_pd = desired_ground_truth_test_pd.loc[predicted_label_desired_ground_truth_jce_test_pd == data_obj.undesired_class]
        legend_handles = create_metric_burden_handles(protected_feat_keys, colors)
        for j in range(len(methods)):
            method = methods[j]
            for prot_feat_idx in range(len(protected_feat_keys)):
                feat = protected_feat_keys[prot_feat_idx]
                feat_unique_val = desired_ground_truth_test_pd[feat].unique()
                len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
                x_pos_list = []
                mean_data_val_list = []
                for feat_idx in range(len(feat_unique_val)):
                    feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_idx],2)]
                    total_ground_truth_feat_val = np.sum(desired_ground_truth_test_pd[feat] == feat_unique_val[feat_idx])
                    total_false_undesired_feat_val = np.sum(false_undesired_test_pd[feat] == feat_unique_val[feat_idx])
                    fnr_feat_val = total_false_undesired_feat_val/total_ground_truth_feat_val
                    x_pos_list.append(fnr_feat_val)
                    feat_method_data = cf_df[(cf_df['cf_method'] == method) & (cf_df.index.isin(idx_feat_values[feat_idx]))]
                    feat_method_data_values = feat_method_data[metric].values
                    mean_data_val_list.append(np.mean(feat_method_data_values))
                    c = colors[prot_feat_idx]
                    if feat_val_name == 'African-American':
                        feat_val_name = 'African'
                    if feat_val_name == 'Non-white':
                        ax[i,j].text(x=fnr_feat_val, y=np.mean(feat_method_data_values), #bbox=dict(ec=c,fc='none'),
                                s=feat_val_name, fontstyle='italic', color=c, size=9, ha='right', va='top')
                    else:
                        ax[i,j].text(x=fnr_feat_val, y=np.mean(feat_method_data_values), #bbox=dict(ec=c,fc='none'),
                                s=feat_val_name, fontstyle='italic', color=c, size=9)
                ax[i,j].scatter(x=x_pos_list, y=mean_data_val_list, color=colors[prot_feat_idx], s=10)
                # ax[i,j].axes.xaxis.set_visible(False)
                ax[i,j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax[i,j].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    for i in range(len(datasets)):
        ax[i,0].set_ylabel(dataset_names[datasets[i]])
        # ax[i,0].legend(handles=legend_handles)
    for j in range(len(methods)):
        ax[0,j].set_title(methods_names[methods[j]])
        ax[-1,j].axes.xaxis.set_visible(True)
    # plt.tight_layout()
    fig.suptitle('$Burden_s$ vs. $FNR_s$')
    plt.subplots_adjust(left=0.11,
                    bottom=0.1, 
                    right=0.95, 
                    top=0.88, 
                    wspace=0.24, 
                    hspace=0.27)
    plt.savefig(results_cf_plots_dir+'fnr_burden.pdf',format='pdf',dpi=400)

def create_handles_awb(colors_dict):
    """
    Method that obtains the accuracy weighted burden for each method and each dataset
    """
    list_handles = []
    for i in range(len(colors_dict.keys())):
        key = list(colors_dict.keys())[i]
        color = colors_dict[key]
        handle = Line2D([0], [0], color=color, lw=2, label=f'{key}')
        list_handles.extend([handle])
    return list_handles

def fnr_plot(datasets, metric, colors_dict):
    """
    Method that obtains the accuracy weighted burden for each method and each dataset
    """
    dataset_names = get_data_names(datasets)
    fig, ax = plt.subplots(nrows=3,ncols=4,sharex=False,sharey=False,figsize=(8,5.5))
    flat_ax = ax.flatten()
    for i in range(len(datasets)):
        data_str = datasets[i]
        eval_obj = load_obj(data_str, 'fnr', 'eval')
        data_obj = load_obj(data_str, 'fnr', 'data')
        model_obj = load_obj(data_str, 'fnr', 'model')
        eval_x_df = eval_obj.all_x_data
        eval_cf_df = eval_obj.all_cf_data
        protected_feat = eval_obj.feat_protected
        protected_feat_keys = list(protected_feat.keys())
        desired_ground_truth_jce_test_pd = data_obj.jce_test_pd.loc[data_obj.test_target != data_obj.undesired_class]
        desired_ground_truth_test_pd = data_obj.test_pd.loc[data_obj.test_target != data_obj.undesired_class]
        predicted_label_desired_ground_truth_jce_test_pd = model_obj.jce_sel.predict(desired_ground_truth_jce_test_pd)
        false_undesired_test_pd = desired_ground_truth_test_pd.loc[predicted_label_desired_ground_truth_jce_test_pd == data_obj.undesired_class]
        fnr_list = []
        feat_list = []
        colors_list = []
        for prot_feat_idx in range(len(protected_feat_keys)):
            feat = protected_feat_keys[prot_feat_idx]
            feat_unique_val = desired_ground_truth_test_pd[feat].unique()
            for feat_idx in range(len(feat_unique_val)):
                feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_idx],2)]
                total_ground_truth_feat_val = np.sum(desired_ground_truth_test_pd[feat] == feat_unique_val[feat_idx])
                # print(f' ground truth {data_str}, {feat}:{feat_val_name}: {total_ground_truth_feat_val} ({total_ground_truth_feat_val/desired_ground_truth_test_pd.shape[0]})')
                # print(f' {data_str}, {feat}:{feat_val_name}: {np.sum(data_obj.test_pd[feat] == feat_unique_val[feat_idx])}')
                total_false_undesired_feat_val = np.sum(false_undesired_test_pd[feat] == feat_unique_val[feat_idx])
                fnr = total_false_undesired_feat_val/total_ground_truth_feat_val
                if feat in ['isMale','isMarried']:
                    feat_val_name = feat+': '+feat_val_name
                fnr_list.append(fnr)
                feat_list.append(feat_val_name)
                colors_list.append(colors_dict[feat_val_name])
        flat_ax[i].bar(x=feat_list,height=fnr_list,color=colors_list)
        flat_ax[i].set_xticklabels(feat_list, rotation = 30, ha='right')
        flat_ax[i].axes.xaxis.set_visible(False)
        flat_ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        flat_ax[i].set_title(dataset_names[datasets[i]])
    legend_handles = create_handles_awb(colors_dict)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.suptitle('False Negative Rate per Sensitive Group ($FNR_s$)')
    fig.legend(loc='lower center', bbox_to_anchor=(0.5,0.00), ncol=6, fancybox=True, shadow=True, handles=legend_handles, prop={'size': 10})
    plt.subplots_adjust(left=0.05,
                    bottom=0.2, 
                    right=0.975, 
                    top=0.91, 
                    wspace=0.25, 
                    hspace=0.22)
    flat_ax[-1].axis('off')
    plt.savefig(results_cf_plots_dir+'fnr.pdf',format='pdf',dpi=400)

def burden_plot(datasets, methods, metric, colors_dict):
    """
    Method that obtains the accuracy weighted burden for each method and each dataset
    """
    methods_names = get_methods_names(methods)
    dataset_names = get_data_names(datasets)
    fig, ax = plt.subplots(nrows=len(datasets),ncols=len(methods),sharex=False,sharey=False,figsize=(8,13))
    for i in range(len(datasets)):
        data_str = datasets[i]
        eval_obj = load_obj(data_str, 'fnr', 'eval')
        data_obj = load_obj(data_str, 'fnr', 'data')
        model_obj = load_obj(data_str, 'fnr', 'model')
        eval_x_df = eval_obj.all_x_data
        eval_cf_df = eval_obj.all_cf_data
        protected_feat = eval_obj.feat_protected
        protected_feat_keys = list(protected_feat.keys())
        x_df, cf_df, original_x_df, original_cf_df = extract_x_cd_df(eval_cf_df, eval_x_df, [metric], data_obj)
        desired_ground_truth_test_pd = data_obj.test_pd.loc[data_obj.test_target != data_obj.undesired_class]
        for j in range(len(methods)):
            method = methods[j]
            awb_list = []
            feat_list = []
            colors_list = []
            for prot_feat_idx in range(len(protected_feat_keys)):   
                feat = protected_feat_keys[prot_feat_idx]
                feat_unique_val = desired_ground_truth_test_pd[feat].unique()
                len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
                for feat_idx in range(len(feat_unique_val)):
                    feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_idx],2)]
                    feat_method_data = cf_df[(cf_df['cf_method'] == method) & (cf_df.index.isin(idx_feat_values[feat_idx]))]
                    burden = np.mean(feat_method_data[metric].values)
                    var = cf_df[cf_df['cf_method'] == method].shape[0]
                    print(f'{data_str}: {var}')
                    if feat in ['isMale','isMarried']:
                        feat_val_name = feat+': '+feat_val_name
                    awb_list.append(burden)
                    feat_list.append(feat_val_name)
                    colors_list.append(colors_dict[feat_val_name])
            ax[i,j].bar(x=feat_list,height=awb_list,color=colors_list)
            ax[i,j].set_xticklabels(feat_list, rotation = 30, ha='right')
            ax[i,j].axes.xaxis.set_visible(False)
            ax[i,j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    legend_handles = create_handles_awb(colors_dict)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    for i in range(len(datasets)):
        ax[i,0].set_ylabel(dataset_names[datasets[i]])
    for j in range(len(methods)):
        ax[0,j].set_title(methods_names[methods[j]])
    fig.suptitle('$Burden_s$')
    fig.legend(loc='lower center', bbox_to_anchor=(0.5,0.00), ncol=6, fancybox=True, shadow=True, handles=legend_handles, prop={'size': 10})
    plt.subplots_adjust(left=0.075,
                    bottom=0.08, 
                    right=0.975, 
                    top=0.94, 
                    wspace=0.25, 
                    hspace=0.05)
    plt.savefig(results_cf_plots_dir+'burden.pdf',format='pdf',dpi=400)

def accuracy_weighted_burden_plot(datasets, methods, metric, colors_dict):
    """
    Method that obtains the accuracy weighted burden for each method and each dataset
    """
    methods_names = get_methods_names(methods)
    dataset_names = get_data_names(datasets)
    fig, ax = plt.subplots(nrows=len(datasets),ncols=len(methods),sharex=False,sharey=False,figsize=(8,13))
    for i in range(len(datasets)):
        data_str = datasets[i]
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
        predicted_label_desired_ground_truth_jce_test_pd = model_obj.jce_sel.predict(desired_ground_truth_jce_test_pd)
        false_undesired_test_pd = desired_ground_truth_test_pd.loc[predicted_label_desired_ground_truth_jce_test_pd == data_obj.undesired_class]
        for j in range(len(methods)):
            method = methods[j]
            awb_list = []
            feat_list = []
            colors_list = []
            for prot_feat_idx in range(len(protected_feat_keys)):
                feat = protected_feat_keys[prot_feat_idx]
                feat_unique_val = desired_ground_truth_test_pd[feat].unique()
                len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
                for feat_idx in range(len(feat_unique_val)):
                    feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_idx],2)]
                    total_ground_truth_feat_val = np.sum(desired_ground_truth_test_pd[feat] == feat_unique_val[feat_idx])
                    total_false_undesired_feat_val = np.sum(false_undesired_test_pd[feat] == feat_unique_val[feat_idx])
                    fnr = total_false_undesired_feat_val/total_ground_truth_feat_val
                    feat_method_data = cf_df[(cf_df['cf_method'] == method) & (cf_df.index.isin(idx_feat_values[feat_idx]))]
                    mean_burden = np.mean(feat_method_data[metric].values)
                    awb = fnr*mean_burden*100/data_obj.test_pd.shape[1]
                    if feat in ['isMale','isMarried']:
                        feat_val_name = feat+': '+feat_val_name
                    awb_list.append(awb)
                    feat_list.append(feat_val_name)
                    colors_list.append(colors_dict[feat_val_name])
            ax[i,j].bar(x=feat_list,height=awb_list,color=colors_list)
            ax[i,j].set_xticklabels(feat_list, rotation = 30, ha='right')
            ax[i,j].axes.xaxis.set_visible(False)
            ax[i,j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    legend_handles = create_handles_awb(colors_dict)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    for i in range(len(datasets)):
        ax[i,0].set_ylabel(dataset_names[datasets[i]])
    for j in range(len(methods)):
        ax[0,j].set_title(methods_names[methods[j]])
    fig.suptitle('Normalized Accuracy Weighted Burden ($NAWB_s$) (%)')
    fig.legend(loc='lower center', bbox_to_anchor=(0.5,0.00), ncol=6, fancybox=True, shadow=True, handles=legend_handles, prop={'size': 10})
    plt.subplots_adjust(left=0.09,
                    bottom=0.08, 
                    right=0.975, 
                    top=0.94, 
                    wspace=0.25, 
                    hspace=0.1)
    plt.savefig(results_cf_plots_dir+'normal_awb.pdf',format='pdf',dpi=400)

datasets = ['compass','law','diabetes']  # Name of the datasets to be analyzed
methods_to_run = ['nn','mo','rt','cchvae'] #['nn','mo','rt','cchvae']
colors_list = ['red', 'blue', 'green', 'purple', 'lightgreen', 'tab:brown', 'orange']

colors_dict = {'Male':'red','Female':'blue','White':'gainsboro','Non-white':'black',
               '<25':'thistle','25-60':'violet','>60':'purple','<18':'green','>=18':'yellow','Single':'peachpuff',
               'Married':'peru','Divorced':'saddlebrown','isMarried: True':'cyan','isMarried: False':'darkcyan',
               'isMale: True':'lightcoral','isMale: False':'firebrick','Other':'lightgreen','HS':'limegreen',
               'University':'green','Graduate':'darkgreen','African-American':'orangered','Caucasian':'coral'}

# attainable_cf_plot(datasets, methods_to_run)
# feature_ratio_change_cf_plot(datasets, methods_to_run)
# method_box_plot(datasets, methods_to_run, 'proximity', colors)
# accuracy_burden_plot(datasets, 'mo', 'proximity', colors)
# statistical_parity_burden_plot(datasets, 'mo', 'proximity', colors)
# equalized_odds_burden_plot(datasets, 'mo', 'proximity', colors)
# fnr_plot(datasets, 'proximity', colors_dict)
# burden_plot(datasets, methods_to_run, 'proximity', colors_dict)
fnr_burden_plot(datasets, methods_to_run, 'proximity', colors_list)
# accuracy_weighted_burden_plot(datasets, methods_to_run, 'proximity', colors_dict)

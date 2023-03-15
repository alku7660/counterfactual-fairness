import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pickle
from support import path_here, results_cf_obj_method_dir, results_cf_plots_dir
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from matplotlib.lines import Line2D
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter
from support import load_obj
matplotlib.rc('ytick', labelsize=9)
matplotlib.rc('xtick', labelsize=9)
# import seaborn as sns
from fairness_clusters import datasets, methods_to_run

def extract_number_idx_instances_feat_val(original_x_df, feat_name, feat_unique_val):
    """
    DESCRIPTION:        Extracts the number of instances per value of a feature of interest

    INPUT:
    original_x_df:      DataFrame containing all the instances of interest in the original format and corresponding accuracy (validity) information
    feat_name:          Name of the feature of interest
    feat_unique_val:    List of unique values of the feature of interest

    OUTPUT:
    len_feat_values:    Number of instances belonging to each feature value
    idx_feat_values:    List of indices of the instances belonging to each feature value
    """
    len_feat_values, idx_feat_values = [], []
    for i in range(len(feat_unique_val)):
        feat_values = original_x_df[original_x_df[feat_name] == feat_unique_val[i]]
        feat_values_idx = feat_values.index.tolist()
        len_feat_values.append(len(feat_values))
        idx_feat_values.append(feat_values_idx)
    return len_feat_values, idx_feat_values

def create_boxplot_handles(protected_feat, original_x_df, color_list):
    """
    DESCRIPTION:                Creates the legend handles for the boxplot subplots for datasets and method calculating the number of examples per sensitive group

    INPUT:
    protected_feat:             Protected features names
    original_x_df:              DataFrame containing the instances of interest
    color_list:                 List of colors to use

    OUTPUT:
    list_handles:               Handles for use on the boxplot subplots
    """
    list_handles = []
    checked_colors = 0
    for feat in protected_feat:
        unique_feat_val = original_x_df[feat].unique()
        for i in range(len(unique_feat_val)):
            feat_val_i = unique_feat_val[i]
            len_feat_val_i = len(original_x_df[original_x_df[feat] == feat_val_i])
            total_instances = len(original_x_df)
            handle = Line2D([0], [0], color=color_list[checked_colors + i], lw=2, label=f'{protected_feat[feat][np.round(feat_val_i, 2)]}: {len_feat_val_i} ({np.round(len_feat_val_i*100/total_instances, 1)}%)')
            list_handles.extend([handle])
        checked_colors += len(unique_feat_val)
    return list_handles

def create_handles_awb(colors_dict, used_features=None):
    """
    DESCRIPTION:            Obtains the accuracy weighted burden for each method and each dataset

    INPUT:
    colors_dict:            Dictionary of colors

    OUTPUT:
    list_handles:           List of handles
    """
    list_handles = []
    for i in range(len(colors_dict.keys())):
        if used_features is None:
            key = list(colors_dict.keys())[i]
            color = colors_dict[key]
            handle = Line2D([0], [0], color=color, lw=2, label=f'{key}')
            list_handles.extend([handle])
        else:
            key = list(colors_dict.keys())[i]
            if key in used_features:
                color = colors_dict[key]
                handle = Line2D([0], [0], color=color, lw=2, label=f'{key} CF')
                list_handles.extend([handle])
            else:
                continue
    return list_handles

def get_data_names(datasets):
    """
    DESCRIPTION:            Gets the names of the datasets for plotting

    INPUT:
    datasets:               List of datasets names

    OUTPUT:
    data_dict:              Dataset dictionary
    """
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
        elif i == 'synthetic_athlete':
            data_dict[i] = 'Athlete'
        elif i == 'synthetic_disease':
            data_dict[i] = 'Disease'
    return data_dict

def get_metric_names(metrics):
    """
    DESCRIPTION:            Gets the names of the performance metrics for plotting

    INPUT:
    metrics:                List of performance metrics to use

    OUTPUT:
    metric_dict:            Performance metrics dictionary
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
    DESCRIPTION:            Gets the names of the methods for plotting

    INPUT:
    metrics:                List of methods to use

    OUTPUT:
    method_dict:            Methods dictionary
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

def attainable_cf_plot(datasets, methods_to_run):
    """
    DESCRIPTION:        Plots the percentage of attainable CFs given feasibility constraints and whether or not to consider feature mutability

    INPUT:              
    datasets:           Datasets names
    methods_to_run:     Methods names

    OUTPUT: (None: plot stored)
    """
    methods_names = get_methods_names(methods_to_run)
    dataset_names = get_data_names(datasets)

def feature_ratio_change_cf_plot(datasets, methods_to_run):
    """
    DESCRIPTION:            Plots the percentage of attainable CFs given feasibility constraints and whether or not to consider feature mutability
    """
    methods_names = get_methods_names(methods_to_run)
    dataset_names = get_data_names(datasets)

def accuracy_burden_plot(datasets, method, metric, colors):
    """
    DESCRIPTION:            Plots the accuracy versus burden for different sensitive groups
    """
    methods_names = get_methods_names([method])
    dataset_names = get_data_names(datasets)

def statistical_parity_burden_plot(datasets, method, metric, colors):
    """
    DESCRIPTION:            Plots the statistical parity versus burden for different sensitive groups
    """
    methods_names = get_methods_names([method])
    dataset_names = get_data_names(datasets)

def equalized_odds_burden_plot(datasets, method, metric, colors):
    """
    DESCRIPTION:            Plots the equalized odds versus burden for different sensitive groups
    """
    methods_names = get_methods_names([method])
    dataset_names = get_data_names(datasets)

def method_box_plot(datasets, methods, metric, colors):
    """
    DESCRIPTION:        Plots the differences w.r.t. the metric of interest among sensitive groups, for all datasets and methods

    INPUT:
    datasets:           Names of the datasets
    methods:            Names of the methods
    metric:             Name of the metric
    colors:             List of colors to be used

    OUTPUT: (None: plot stored)
    """
    methods_names = get_methods_names(methods)
    dataset_names = get_data_names(datasets)
    metric_names = get_metric_names([metric])
    fig, ax = plt.subplots(nrows=len(datasets), ncols=len(methods), sharex=False, sharey=False, figsize=(8,13))
    for dataset_idx in range(len(datasets)):
        data_str = datasets[dataset_idx]
        for method_idx in range(len(methods)):
            method_str = methods[method_idx]
            eval_obj = load_obj(f'{data_str}_{method_str}_eval.pkl')
            protected_feat = eval_obj.feat_protected
            protected_feat_keys = list(protected_feat.keys())
            original_x_df = pd.concat(eval_obj.original_x.values(), axis=0)
            metrics_cf_df = pd.concat((pd.DataFrame.from_dict(eval_obj.cf_proximity, orient='index', columns=['proximity']), 
                                      pd.DataFrame.from_dict(eval_obj.cf_sparsity, orient='index', columns=['sparsity']), 
                                      pd.DataFrame.from_dict(eval_obj.cf_feasibility, orient='index', columns=['feasibility']), 
                                      pd.DataFrame.from_dict(eval_obj.cf_time, orient='index', columns=['time'])), axis=1)
            sum_feat_unique_val = 0
            feat_val_labels = []
            for feat in protected_feat_keys:
                feat_unique_val = original_x_df[feat].unique()
                len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
                xaxis_pos_box = np.arange(sum_feat_unique_val+len(feat_unique_val))
                for feat_val_idx in range(len(feat_unique_val)):
                    feat_val_instances_idx = idx_feat_values[feat_val_idx]
                    box_feat_val_pos = xaxis_pos_box[sum_feat_unique_val+feat_val_idx]
                    feat_method_data_values = metrics_cf_df.loc[feat_val_instances_idx, metric].values
                    c = colors[sum_feat_unique_val+feat_val_idx]
                    ax[dataset_idx, method_idx].boxplot(x=feat_method_data_values, positions=[box_feat_val_pos], boxprops=dict(color=c),
                            capprops=dict(color=c), showfliers=False, whiskerprops=dict(color=c),
                            medianprops=dict(color=c), widths=0.9, showmeans=True,
                            meanprops=dict(markerfacecolor=c, markeredgecolor=c, marker='D'), flierprops=dict(markeredgecolor=c), notch=False)
                sum_feat_unique_val += len(feat_unique_val)
                feat_val_labels.append(feat_unique_val)
            if method_idx == 0:
                legend_elements = create_boxplot_handles(protected_feat, original_x_df, colors_list)
                ax[dataset_idx, method_idx].legend(handles=legend_elements)
            ax[dataset_idx, method_idx].axes.xaxis.set_visible(False)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    for i in range(len(datasets)):
        ax[i,0].set_ylabel(dataset_names[datasets[i]])
    for j in range(len(methods)):
        ax[0,j].set_title(methods_names[methods[j]])
    fig.suptitle(metric_names[metric])
    fig.legend(handles=legend_elements) #loc=(-0.1,-0.1*len(legend_elements))
    plt.tight_layout()
    plt.savefig(results_cf_plots_dir+f'{metric}_comparison_across_dataset_method.pdf')

def fnr_plot(datasets, colors_dict):
    """
    DESCRIPTION:        Obtains the false negative rate plots for each sensitive feature

    INPUT:
    datasets:           Names of the datasets
    colors_dict:        Dictionary of colors to be used per feature

    OUTPUT: (None: plot stored)
    """
    dataset_names = get_data_names(datasets)
    fig, ax = plt.subplots(nrows=3,ncols=4,sharex=False,sharey=False,figsize=(8,5.5))
    flat_ax = ax.flatten()
    for dataset_idx in range(len(datasets)):
        data_str = datasets[dataset_idx]
        eval_obj = load_obj(f'{data_str}_nn_eval.pkl')
        protected_feat = eval_obj.feat_protected
        protected_feat_keys = list(protected_feat.keys())
        fnr_list = []
        feat_list = []
        colors_list = []
        for feat_idx in range(len(protected_feat_keys)):
            feat = protected_feat_keys[feat_idx]
            feat_unique_val = eval_obj.desired_ground_truth_test_df[feat].unique()
            for feat_val_idx in range(len(feat_unique_val)):
                feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_val_idx],2)]
                total_ground_truth_feat_val = np.sum(eval_obj.desired_ground_truth_test_df[feat] == feat_unique_val[feat_val_idx])
                total_false_undesired_feat_val = np.sum(eval_obj.false_undesired_test_df[feat] == feat_unique_val[feat_val_idx])
                fnr = total_false_undesired_feat_val/total_ground_truth_feat_val
                if feat in ['isMale','isMarried']:
                    feat_val_name = feat+': '+feat_val_name
                fnr_list.append(fnr)
                feat_list.append(feat_val_name)
                colors_list.append(colors_dict[feat_val_name])
        flat_ax[dataset_idx].bar(x=feat_list, height=fnr_list, color=colors_list)
        flat_ax[dataset_idx].set_xticklabels(feat_list, rotation = 30, ha='right')
        flat_ax[dataset_idx].axes.xaxis.set_visible(False)
        flat_ax[dataset_idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        flat_ax[dataset_idx].set_title(dataset_names[datasets[dataset_idx]])
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

def burden_plot(datasets, methods, colors_dict):
    """
    DESCRIPTION:        Obtains the burden for each method and each dataset

    INPUT:
    datasets:           Names of the datasets
    methods:            Names of the methods
    colors_dict:        Dictionary of colors to be used per feature

    OUTPUT: (None: plot stored)
    """
    methods_names = get_methods_names(methods)
    dataset_names = get_data_names(datasets)
    fig, ax = plt.subplots(nrows=len(datasets), ncols=len(methods), sharex=False, sharey=False, figsize=(8,13))
    for dataset_idx in range(len(datasets)):
        data_str = datasets[dataset_idx]
        for method_idx in range(len(methods)):
            method_str = methods[method_idx]
            eval_obj = load_obj(f'{data_str}_{method_str}_eval.pkl')
            protected_feat = eval_obj.feat_protected
            protected_feat_keys = list(protected_feat.keys())
            original_x_df = pd.concat(eval_obj.original_x.values(), axis=0)
            proximity_df = pd.DataFrame.from_dict(eval_obj.cf_proximity, orient='index', columns=['proximity'])
            awb_list = []
            feat_list = []
            colors_list = []
            for feat_idx in range(len(protected_feat_keys)):   
                feat = protected_feat_keys[feat_idx]
                feat_unique_val = eval_obj.desired_ground_truth_test_df[feat].unique()
                len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
                for feat_val_idx in range(len(feat_unique_val)):
                    feat_val_instances_idx = idx_feat_values[feat_val_idx]
                    feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_val_idx],2)]
                    feat_method_data = proximity_df.loc[feat_val_instances_idx, 'proximity'].values
                    burden = np.mean(feat_method_data)
                    if feat in ['isMale','isMarried']:
                        feat_val_name = feat+': '+feat_val_name
                    awb_list.append(burden)
                    feat_list.append(feat_val_name)
                    colors_list.append(colors_dict[feat_val_name])
            ax[dataset_idx, method_idx].bar(x=feat_list,height=awb_list,color=colors_list)
            ax[dataset_idx, method_idx].set_xticklabels(feat_list, rotation = 30, ha='right')
            ax[dataset_idx, method_idx].axes.xaxis.set_visible(False)
            ax[dataset_idx, method_idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
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
    plt.savefig(results_cf_plots_dir+'burden.pdf',format='pdf')

def fnr_burden_plot(datasets, methods, colors):
    """
    DESCRIPTION:        Obtains false negative rate plots for each sensitive feature and compares it with the burden of each sensitive group

    INPUT:
    datasets:           Names of the datasets
    methods:            Names of the methods
    colors_dict:        Dictionary of colors to be used per feature

    OUTPUT: (None: plot stored) 
    """
    methods_names = get_methods_names(methods)
    dataset_names = get_data_names(datasets)
    fig, ax = plt.subplots(nrows=len(datasets), ncols=len(methods), sharex=False, sharey=False, figsize=(8,4.5))
    fig.supxlabel('$FNR_s$')
    fig.supylabel('$Burden_s$ (Lower is Better)')
    for dataset_idx in range(len(datasets)):
        data_str = datasets[dataset_idx]
        for method_idx in range(len(methods)):
            method_str = methods[method_idx]
            eval_obj = load_obj(f'{data_str}_{method_str}_eval.pkl')
            protected_feat = eval_obj.feat_protected
            protected_feat_keys = list(protected_feat.keys())
            original_x_df = pd.concat(eval_obj.original_x.values(), axis=0)
            proximity_df = pd.DataFrame.from_dict(eval_obj.cf_proximity, orient='index', columns=['proximity'])
            for feat_idx in range(len(protected_feat_keys)):
                feat = protected_feat_keys[feat_idx]
                feat_unique_val = eval_obj.desired_ground_truth_test_df[feat].unique()
                len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
                x_pos_list = []
                mean_data_val_list = []
                for feat_val_idx in range(len(feat_unique_val)):
                    feat_val_instances_idx = idx_feat_values[feat_val_idx]
                    feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_val_idx],2)]
                    total_ground_truth_feat_val = np.sum(eval_obj.desired_ground_truth_test_df[feat] == feat_unique_val[feat_val_idx])
                    total_false_undesired_feat_val = np.sum(eval_obj.false_undesired_test_df[feat] == feat_unique_val[feat_val_idx])
                    fnr_feat_val = total_false_undesired_feat_val/total_ground_truth_feat_val
                    x_pos_list.append(fnr_feat_val)
                    feat_method_data_values = proximity_df.loc[feat_val_instances_idx, 'proximity'].values
                    mean_data_val_list.append(np.mean(feat_method_data_values))
                    if feat_val_name == 'African-American':
                        feat_val_name = 'Afric.'
                    if feat_val_name == 'Non-white':
                        feat_val_name = 'Non-w.'
                    ax[dataset_idx, method_idx].text(x=fnr_feat_val, y=np.mean(feat_method_data_values), #bbox=dict(ec=c,fc='none'),
                                s=feat_val_name, fontstyle='italic', color=colors[feat_idx], size=9)
                    ax[dataset_idx, method_idx].scatter(x=x_pos_list, y=mean_data_val_list, color=colors[feat_idx], s=10)
                ax[dataset_idx, method_idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax[dataset_idx, method_idx].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    for i in range(len(datasets)):
        ax[i,0].set_ylabel(dataset_names[datasets[i]])
    for j in range(len(methods)):
        ax[0,j].set_title(methods_names[methods[j]])
        ax[-1,j].axes.xaxis.set_visible(True)
    fig.suptitle('$Burden_s$ vs. $FNR_s$')
    plt.subplots_adjust(left=0.11,
                    bottom=0.1, 
                    right=0.95, 
                    top=0.88, 
                    wspace=0.24, 
                    hspace=0.27)
    plt.savefig(results_cf_plots_dir+'fnr_burden.pdf',format='pdf',dpi=400)

def nawb_plot(datasets, methods, colors_dict):
    """
    DESCRIPTION:        Obtains the accuracy weighted burden for each method and each dataset

    INPUT:
    datasets:           Names of the datasets
    methods:            Names of the methods
    colors_dict:        Dictionary of colors to be used per feature

    OUTPUT: (None: plot stored)
    """
    methods_names = get_methods_names(methods)
    dataset_names = get_data_names(datasets)
    fig, ax = plt.subplots(nrows=len(datasets), ncols=len(methods), sharex=False, sharey=False, figsize=(8,13))
    for dataset_idx in range(len(datasets)):
        data_str = datasets[dataset_idx]
        for method_idx in range(len(methods)):
            method_str = methods[method_idx]
            eval_obj = load_obj(f'{data_str}_{method_str}_eval.pkl')
            protected_feat = eval_obj.feat_protected
            protected_feat_keys = list(protected_feat.keys())
            original_x_df = pd.concat(eval_obj.original_x.values(), axis=0)
            proximity_df = pd.DataFrame.from_dict(eval_obj.cf_proximity, orient='index', columns=['proximity'])
            nawb_list = []
            feat_list = []
            colors_list = []
            for feat_idx in range(len(protected_feat_keys)):
                feat = protected_feat_keys[feat_idx]
                feat_unique_val = eval_obj.desired_ground_truth_test_df[feat].unique()
                len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
                for feat_val_idx in range(len(feat_unique_val)):
                    feat_val_instances_idx = idx_feat_values[feat_val_idx]
                    feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_val_idx],2)]
                    total_ground_truth_feat_val = np.sum(eval_obj.desired_ground_truth_test_df[feat] == feat_unique_val[feat_val_idx])
                    total_false_undesired_feat_val = np.sum(eval_obj.false_undesired_test_df[feat] == feat_unique_val[feat_val_idx])
                    fnr = total_false_undesired_feat_val/total_ground_truth_feat_val
                    feat_method_data_values = proximity_df.loc[feat_val_instances_idx, 'proximity'].values
                    mean_burden = np.mean(feat_method_data_values)
                    nawb = fnr*mean_burden*100/len(eval_obj.data_cols)
                    if feat in ['isMale','isMarried']:
                        feat_val_name = feat+': '+feat_val_name
                    nawb_list.append(nawb)
                    feat_list.append(feat_val_name)
                    colors_list.append(colors_dict[feat_val_name])
            ax[dataset_idx, method_idx].bar(x=feat_list,height=nawb_list,color=colors_list)
            ax[dataset_idx, method_idx].set_xticklabels(feat_list, rotation = 30, ha='right')
            ax[dataset_idx, method_idx].axes.xaxis.set_visible(False)
            ax[dataset_idx, method_idx].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
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

def validity_groups_cf(datasets, methods):
    """
    DESCRIPTION:        Obtains the validity percentage of all groups counterfactuals

    INPUT:
    datasets:           Names of the datasets
    methods:            Names of the methods

    OUTPUT: (None: plot stored)
    """
    dataset_names = get_data_names(datasets)
    methods_names = get_methods_names(methods)
    fig, ax = plt.subplots(nrows=len(datasets), ncols=1, sharex=False, sharey=False, figsize=(4,10))
    for dataset_idx in range(len(datasets)):
        data_str = datasets[dataset_idx]
        for method_idx in range(len(methods)):
            method_str = methods[method_idx]
            method_name = methods_names[methods[method_idx]]
            eval_obj = load_obj(f'{data_str}_{method_str}_eval.pkl')
            group_cf_validity_df = pd.DataFrame.from_dict(eval_obj.group_cf_validity, orient='index', columns=[method_name])
            if method_idx == 0:
                validity_df = group_cf_validity_df
            else:
                validity_df = pd.concat((validity_df, group_cf_validity_df), axis=1)
        sns.heatmap(validity_df, cbar=True, annot=True, ax=ax[dataset_idx])
    for i in range(len(datasets)):
        ax[i].set_ylabel(dataset_names[datasets[i]])
    fig.suptitle('Group Counterfactual Validity')
    plt.subplots_adjust(left=0.3,
                    bottom=0.05, 
                    right=0.8, 
                    top=0.95, 
                    wspace=0.1, 
                    hspace=0.2)
    plt.savefig(results_cf_plots_dir+'group_cf_validity.pdf',format='pdf')

def validity_clusters(datasets, methods):
    """
    DESCRIPTION:        Obtains the validity percentage of all clusters

    INPUT:
    datasets:           Names of the datasets
    methods:            Names of the methods

    OUTPUT: (None: plot stored)
    """
    dataset_names = get_data_names(datasets)
    methods_names = get_methods_names(methods)
    fig, ax = plt.subplots(nrows=len(datasets), ncols=1, sharex=False, sharey=False, figsize=(4,11))
    for dataset_idx in range(len(datasets)):
        data_str = datasets[dataset_idx]
        for method_idx in range(len(methods)):
            method_str = methods[method_idx]
            method_name = methods_names[methods[method_idx]]
            eval_obj = load_obj(f'{data_str}_{method_str}_eval.pkl')
            cluster_validity_df = pd.DataFrame.from_dict(eval_obj.cluster_validity, orient='index', columns=[method_name])
            if method_idx == 0:
                validity_df = cluster_validity_df
            else:
                validity_df = pd.concat((validity_df, cluster_validity_df), axis=1)
        sns.heatmap(validity_df, cbar=True, annot=True, ax=ax[dataset_idx])
    for i in range(len(datasets)):
        ax[i].set_ylabel(dataset_names[datasets[i]])
    fig.suptitle('Cluster Validity')
    plt.subplots_adjust(left=0.3,
                    bottom=0.05, 
                    right=0.8, 
                    top=0.95, 
                    wspace=0.1, 
                    hspace=0.2)
    plt.savefig(results_cf_plots_dir+'cluster_validity.pdf',format='pdf')

def burden_groups_cf(datasets, methods):
        """
        DESCRIPTION:        Obtains the burden for each sensitive group w.r.t. each of the group counterfactuals

        INPUT:
        datasets:           Names of the datasets
        methods:            Names of the methods

        OUTPUT: (None: plot stored)
        """
        methods_names = get_methods_names(methods)
        dataset_names = get_data_names(datasets)
        fig, ax = plt.subplots(nrows=len(datasets), ncols=len(methods), sharex=False, sharey=False, figsize=(8,13))
        for dataset_idx in range(len(datasets)):
            data_str = datasets[dataset_idx]
            for method_idx in range(len(methods)):
                method_str = methods[method_idx]
                eval_obj = load_obj(f'{data_str}_{method_str}_eval.pkl')
                protected_feat = eval_obj.feat_protected
                protected_feat_keys = list(protected_feat.keys())
                original_x_df = pd.concat(eval_obj.original_x.values(), axis=0)
                group_cf_proximity = eval_obj.group_cf_proximity
                groups_names = group_cf_proximity.columns
                burden = pd.DataFrame(index=groups_names, columns=groups_names)
                feat_list = []
                for feat_idx in range(len(protected_feat_keys)):   
                    feat = protected_feat_keys[feat_idx]
                    feat_unique_val = original_x_df[feat].unique()
                    len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
                    for feat_val_idx in range(len(feat_unique_val)):
                        feat_val_instances_idx = idx_feat_values[feat_val_idx]
                        feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_val_idx],2)]
                        for group in groups_names:
                            feat_method_data = group_cf_proximity.loc[feat_val_instances_idx, group].values
                            burden.loc[feat_val_name, group] = np.mean(feat_method_data)
                        if feat in ['isMale','isMarried']:
                            feat_val_name = feat+': '+feat_val_name
                        feat_list.append(feat_val_name)
                for group in groups_names:
                    burden.loc['all', group] = np.mean(group_cf_proximity.loc[:, group].values)
                burden = burden.apply(pd.to_numeric)
                sns.heatmap(burden, cbar=True, annot=True, fmt='.3f', ax=ax[dataset_idx, method_idx])
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        for i in range(len(datasets)):
            ax[i,0].set_ylabel(dataset_names[datasets[i]])
        for j in range(len(methods)):
            ax[0,j].set_title(methods_names[methods[j]])
        fig.suptitle('$Burden_{s}$ for Group Counterfactuals')
        plt.subplots_adjust(left=0.075,
                        bottom=0.08, 
                        right=0.975, 
                        top=0.94, 
                        wspace=0.2, 
                        hspace=0.2)
        plt.savefig(results_cf_plots_dir+'group_cf_burden_instances.pdf',format='pdf')

def burden_groups_cf_bar(datasets, method_str):
        """
        DESCRIPTION:        Obtains the burden for each sensitive group w.r.t. each of the group counterfactuals

        INPUT:
        datasets:           Names of the datasets
        methods:            Names of the methods

        OUTPUT: (None: plot stored)
        """
        dataset_names = get_data_names(datasets)
        fig, ax = plt.subplots(nrows=len(datasets), ncols=1, sharex=False, sharey=False, figsize=(7,10))
        used_features = []
        for dataset_idx in range(len(datasets)):
            width = 0.1
            multiplier = 0
            data_str = datasets[dataset_idx]
            eval_obj = load_obj(f'{data_str}_{method_str}_eval.pkl')
            protected_feat = eval_obj.feat_protected
            protected_feat_keys = list(protected_feat.keys())
            original_x_df = pd.concat(eval_obj.original_x.values(), axis=0)
            group_cf_proximity = eval_obj.group_cf_proximity
            groups_names = list(group_cf_proximity.columns)
            burden_mean = pd.DataFrame(index=groups_names, columns=groups_names)
            burden_std = pd.DataFrame(index=groups_names, columns=groups_names)
            feat_list = []
            for feat_idx in range(len(protected_feat_keys)):   
                feat = protected_feat_keys[feat_idx]
                feat_unique_val = original_x_df[feat].unique()
                _, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
                for feat_val_idx in range(len(feat_unique_val)):
                    feat_val_instances_idx = idx_feat_values[feat_val_idx]
                    feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_val_idx],2)]
                    for group in groups_names:
                        feat_method_data = group_cf_proximity.loc[feat_val_instances_idx, group].values
                        burden_mean.loc[feat_val_name, group] = np.mean(feat_method_data)
                        burden_std.loc[feat_val_name, group] = np.std(feat_method_data, ddof=1)
                    if feat in ['isMale','isMarried']:
                        feat_val_name = feat+': '+feat_val_name
                    feat_list.append(feat_val_name)
            for group in groups_names:
                burden_mean.loc['all', group] = np.mean(group_cf_proximity.loc[:, group].values)
                burden_std.loc['all', group] = np.std(group_cf_proximity.loc[:, group].values, ddof=1)
            burden_mean = burden_mean.apply(pd.to_numeric)
            burden_std = burden_std.apply(pd.to_numeric)
            x = np.arange(len(groups_names))
            for col in burden_mean.columns:
                offset = width*multiplier
                graph = ax[dataset_idx].bar(x + offset, burden_mean.loc[:, col], width, label=col, color=colors_dict[col.capitalize()]) # yerr=burden_std.loc[idx, :]
                # ax[dataset_idx].bar_label(graph, fmt='%.3f', padding=3)
                # ax[dataset_idx].legend(loc='upper left', ncol=len(burden_mean.index))
                ax[dataset_idx].set_xticks(x + width*0.5*(len(x) - 1), [f'{i.capitalize()} inst.' for i in groups_names])
                ax[dataset_idx].set_ylabel(dataset_names[datasets[dataset_idx]])
                multiplier += 1
                used_features.append(col.capitalize())
        legend_handles = create_handles_awb(colors_dict, used_features)
        fig.legend(loc='lower center', bbox_to_anchor=(0.5,0.025), ncol=4, fancybox=True, shadow=True, handles=legend_handles, prop={'size': 10})
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        fig.suptitle(f'{method_str.upper()} $Burden_s$ for Group Counterfactuals')
        fig.supxlabel(f'Sensitive Group Instances')
        fig.supylabel(f'Avg. Burden')
        plt.subplots_adjust(left=0.125,
                        bottom=0.1, 
                        right=0.975, 
                        top=0.94, 
                        wspace=0.2, 
                        hspace=0.2)
        plt.savefig(f'{results_cf_plots_dir}{method_str.upper()}_group_cf_burden_instances_bar.pdf',format='pdf')

def burden_cluster_cf(datasets, methods):
        """
        DESCRIPTION:        Obtains the burden for each sensitive groups cluster w.r.t. each of the cluster counterfactuals

        INPUT:
        datasets:           Names of the datasets
        methods:            Names of the methods

        OUTPUT: (None: plot stored)
        """
        methods_names = get_methods_names(methods)
        dataset_names = get_data_names(datasets)
        fig, ax = plt.subplots(nrows=len(datasets), ncols=len(methods), sharex=False, sharey=False, figsize=(8,13))
        for dataset_idx in range(len(datasets)):
            data_str = datasets[dataset_idx]
            for method_idx in range(len(methods)):
                method_str = methods[method_idx]
                eval_obj = load_obj(f'{data_str}_{method_str}_eval.pkl')
                protected_feat = eval_obj.feat_protected
                protected_feat_keys = list(protected_feat.keys())
                original_x_df = pd.concat(eval_obj.original_x.values(), axis=0)
                cluster_cf_proximity = eval_obj.cluster_cf_proximity
                groups_names = cluster_cf_proximity.columns
                burden = pd.DataFrame(index=groups_names, columns=groups_names)
                feat_list = []
                for feat_idx in range(len(protected_feat_keys)):   
                    feat = protected_feat_keys[feat_idx]
                    feat_unique_val = original_x_df[feat].unique()
                    len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
                    for feat_val_idx in range(len(feat_unique_val)):
                        feat_val_instances_idx = idx_feat_values[feat_val_idx]
                        feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_val_idx],2)]
                        for group in groups_names:
                            feat_method_data = cluster_cf_proximity.loc[feat_val_instances_idx, group].values
                            burden.loc[feat_val_name, group] = np.mean(feat_method_data)
                        if feat in ['isMale','isMarried']:
                            feat_val_name = feat+': '+feat_val_name
                        feat_list.append(feat_val_name)
                for group in groups_names:
                    burden.loc['all', group] = np.mean(cluster_cf_proximity.loc[:, group].values)
                burden = burden.apply(pd.to_numeric)
                sns.heatmap(burden, cbar=True, annot=True, fmt='.3f', ax=ax[dataset_idx, method_idx])
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        for i in range(len(datasets)):
            ax[i,0].set_ylabel(dataset_names[datasets[i]])
        for j in range(len(methods)):
            ax[0,j].set_title(methods_names[methods[j]])
        fig.suptitle('$Burden_s$ for Cluster Counterfactuals')
        plt.subplots_adjust(left=0.075,
                        bottom=0.08, 
                        right=0.975, 
                        top=0.94, 
                        wspace=0.2, 
                        hspace=0.2)
        plt.savefig(results_cf_plots_dir+'cluster_cf_burden_instances.pdf',format='pdf')

def nawb_groups_cf(datasets, methods):
        """
        DESCRIPTION:        Obtains the NAWB measure for each sensitive group w.r.t. each of the group counterfactuals

        INPUT:
        datasets:           Names of the datasets
        methods:            Names of the methods

        OUTPUT: (None: plot stored)
        """
        methods_names = get_methods_names(methods)
        dataset_names = get_data_names(datasets)
        fig, ax = plt.subplots(nrows=len(datasets), ncols=len(methods), sharex=False, sharey=False, figsize=(8,13))
        for dataset_idx in range(len(datasets)):
            data_str = datasets[dataset_idx]
            for method_idx in range(len(methods)):
                method_str = methods[method_idx]
                eval_obj = load_obj(f'{data_str}_{method_str}_eval.pkl')
                protected_feat = eval_obj.feat_protected
                protected_feat_keys = list(protected_feat.keys())
                original_x_df = pd.concat(eval_obj.original_x.values(), axis=0)
                group_cf_proximity = eval_obj.group_cf_proximity
                groups_names = group_cf_proximity.columns
                nawb = pd.DataFrame(index=groups_names, columns=groups_names)
                feat_list = []
                for feat_idx in range(len(protected_feat_keys)):   
                    feat = protected_feat_keys[feat_idx]
                    feat_unique_val = original_x_df[feat].unique()
                    len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
                    for feat_val_idx in range(len(feat_unique_val)):
                        feat_val_instances_idx = idx_feat_values[feat_val_idx]
                        feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_val_idx],2)]
                        total_ground_truth_feat_val = np.sum(eval_obj.desired_ground_truth_test_df[feat] == feat_unique_val[feat_val_idx])
                        total_false_undesired_feat_val = np.sum(eval_obj.false_undesired_test_df[feat] == feat_unique_val[feat_val_idx])
                        fnr_group = total_false_undesired_feat_val/total_ground_truth_feat_val
                        for group in groups_names:
                            feat_method_data = group_cf_proximity.loc[feat_val_instances_idx, group].values
                            mean_burden = np.mean(feat_method_data)
                            nawb.loc[feat_val_name, group] = fnr_group*mean_burden*100/len(eval_obj.data_cols)
                        if feat in ['isMale','isMarried']:
                            feat_val_name = feat+': '+feat_val_name
                        feat_list.append(feat_val_name)
                for group in groups_names:
                    nawb.loc['all', group] = np.mean(group_cf_proximity.loc[:, group].values)
                nawb = nawb.apply(pd.to_numeric)
                sns.heatmap(nawb, cbar=True, annot=True, fmt='.3f', ax=ax[dataset_idx, method_idx])
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        for i in range(len(datasets)):
            ax[i,0].set_ylabel(dataset_names[datasets[i]])
        for j in range(len(methods)):
            ax[0,j].set_title(methods_names[methods[j]])
        fig.suptitle('$NAWB_s$ for Group Counterfactuals')
        plt.subplots_adjust(left=0.075,
                        bottom=0.08, 
                        right=0.975, 
                        top=0.94, 
                        wspace=0.2, 
                        hspace=0.2)
        plt.savefig(results_cf_plots_dir+'group_cf_nawb_instances.pdf',format='pdf')

def nawb_cluster_cf(datasets, methods):
        """
        DESCRIPTION:        Obtains the NAWB measure for each sensitive group cluster w.r.t. each of the cluster counterfactuals

        INPUT:
        datasets:           Names of the datasets
        methods:            Names of the methods

        OUTPUT: (None: plot stored)
        """
        methods_names = get_methods_names(methods)
        dataset_names = get_data_names(datasets)
        fig, ax = plt.subplots(nrows=len(datasets), ncols=len(methods), sharex=False, sharey=False, figsize=(8,13))
        for dataset_idx in range(len(datasets)):
            data_str = datasets[dataset_idx]
            for method_idx in range(len(methods)):
                method_str = methods[method_idx]
                eval_obj = load_obj(f'{data_str}_{method_str}_eval.pkl')
                protected_feat = eval_obj.feat_protected
                protected_feat_keys = list(protected_feat.keys())
                original_x_df = pd.concat(eval_obj.original_x.values(), axis=0)
                cluster_cf_proximity = eval_obj.cluster_cf_proximity
                groups_names = cluster_cf_proximity.columns
                nawb = pd.DataFrame(index=groups_names, columns=groups_names)
                feat_list = []
                for feat_idx in range(len(protected_feat_keys)):   
                    feat = protected_feat_keys[feat_idx]
                    feat_unique_val = original_x_df[feat].unique()
                    len_feat_values, idx_feat_values = extract_number_idx_instances_feat_val(original_x_df, feat, feat_unique_val)
                    for feat_val_idx in range(len(feat_unique_val)):
                        feat_val_instances_idx = idx_feat_values[feat_val_idx]
                        feat_val_name = protected_feat[feat][np.round(feat_unique_val[feat_val_idx],2)]
                        total_ground_truth_feat_val = np.sum(eval_obj.desired_ground_truth_test_df[feat] == feat_unique_val[feat_val_idx])
                        total_false_undesired_feat_val = np.sum(eval_obj.false_undesired_test_df[feat] == feat_unique_val[feat_val_idx])
                        fnr_group = total_false_undesired_feat_val/total_ground_truth_feat_val
                        for group in groups_names:
                            feat_method_data = cluster_cf_proximity.loc[feat_val_instances_idx, group].values
                            mean_burden = np.mean(feat_method_data)
                            nawb.loc[feat_val_name, group] = fnr_group*mean_burden*100/len(eval_obj.data_cols)
                        if feat in ['isMale','isMarried']:
                            feat_val_name = feat+': '+feat_val_name
                        feat_list.append(feat_val_name)
                for group in groups_names:
                    nawb.loc['all', group] = np.mean(cluster_cf_proximity.loc[:, group].values)
                nawb = nawb.apply(pd.to_numeric)
                sns.heatmap(nawb, cbar=True, annot=True, fmt='.3f', ax=ax[dataset_idx, method_idx])
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        for i in range(len(datasets)):
            ax[i,0].set_ylabel(dataset_names[datasets[i]])
        for j in range(len(methods)):
            ax[0,j].set_title(methods_names[methods[j]])
        fig.suptitle('$NAWB_s$ for Cluster Counterfactuals')
        plt.subplots_adjust(left=0.075,
                        bottom=0.08, 
                        right=0.975, 
                        top=0.94, 
                        wspace=0.2, 
                        hspace=0.2)
        plt.savefig(results_cf_plots_dir+'cluster_cf_nawb_instances.pdf',format='pdf')

def inverse_transform_only(bin_enc, cat_enc, bin_enc_cols, cat_enc_cols, binary, categorical, numerical, instance):
    """
    DESCRIPTION:            Transforms an instance to the original features
    
    INPUT:
    instance:               Instance of interest

    OUTPUT:
    original_instance_df:   Instance of interest in the original feature format
    """
    instance_index = instance.index
    original_instance_df = pd.DataFrame(index=instance_index)
    if len(bin_enc_cols) > 0:
        instance_bin = bin_enc.inverse_transform(instance[bin_enc_cols])
        instance_bin_pd = pd.DataFrame(data=instance_bin, index=instance_index, columns=binary)
        original_instance_df = pd.concat((original_instance_df, instance_bin_pd), axis=1)
    if len(cat_enc_cols) > 0:
        instance_cat = cat_enc.inverse_transform(instance[cat_enc_cols])
        instance_cat_pd = pd.DataFrame(data=instance_cat, index=instance_index, columns=categorical)
        original_instance_df = pd.concat((original_instance_df, instance_cat_pd), axis=1)
    if len(numerical) > 0:
        instance_num = instance[numerical]
        instance_num_pd = pd.DataFrame(data=instance_num, index=instance_index, columns=numerical)
        original_instance_df = pd.concat((original_instance_df, instance_num_pd), axis=1)
    return original_instance_df

def plot_centroids():
    """
    Plot all centroids of clusters found for each feature and feature_value
    """
    method_str = 'nn'
    for data_str in datasets:
        eval_obj = load_obj(f'{data_str}_{method_str}_eval.pkl')
        clusters = eval_obj.cluster_obj
        cluster_centroid_dict = clusters.centroids
        cluster_instance_dict = clusters.clusters
        cluster_centroid_feat_list = list(cluster_centroid_dict.keys())
        features = eval_obj.binary + eval_obj.categorical + eval_obj.numerical
        for feat in cluster_centroid_feat_list:
            feat_val_list = list(cluster_centroid_dict[feat].keys())
            fig, ax = plt.subplots(nrows=1, ncols=len(feat_val_list))
            for feat_val_idx in range(len(feat_val_list)):
                feat_val = feat_val_list[feat_val_idx]
                centroid_list = cluster_centroid_dict[feat][feat_val]
                instance_list = cluster_instance_dict[feat][feat_val]
                for idx in range(len(centroid_list)):
                    bin_enc, cat_enc, bin_enc_cols, cat_enc_cols, binary, categorical, numerical = eval_obj.bin_enc, eval_obj.cat_enc, eval_obj.bin_enc_cols, eval_obj.cat_enc_cols, eval_obj.binary, eval_obj.categorical, eval_obj.numerical
                    centroid_original = inverse_transform_only(bin_enc, cat_enc, bin_enc_cols, cat_enc_cols, binary, categorical, numerical, centroid_list[idx]).values[0]
                    ax[feat_val_idx].plot(features, centroid_original, color=colors_list[feat_val_idx])
                ax[feat_val_idx].set_xlabel('Features')
                ax[feat_val_idx].set_ylabel('Values')
                ax[feat_val_idx].set_title(f'{feat}: {feat_val} (#: {int(np.sum([len(instance_list[i]) for i in range(len(instance_list))]))} C: {len(centroid_list)})', wrap=True)
                ax[feat_val_idx].set_xticklabels(features, rotation=45, ha='right')
                # legend_list = [Line2D([0], [0], color=colors_list[feat_val_idx_plot], lw=3) for feat_val_idx_plot in range(len(feat_val_list))]
                # ax[feat_val_idx].legend(legend_list, feat_val_list)
            plt.savefig(f'{results_cf_plots_dir}{data_str}_{feat}_centroids.pdf', format='pdf')

def plot_centroids_cfs():
    """
    Plots the centroids with their corresponding counterfactuals
    """
    for data_str in datasets:
        for method_idx in range(len(methods_to_run)):
            method_str = methods_to_run[method_idx]
            eval_obj = load_obj(f'{data_str}_{method_str}_eval.pkl')
            clusters = eval_obj.cluster_obj
            cluster_centroid_dict = clusters.centroids
            cluster_instance_dict = clusters.clusters
            cluster_centroid_feat_list = list(cluster_centroid_dict.keys())
            features = eval_obj.binary + eval_obj.categorical + eval_obj.numerical
            for feat in cluster_centroid_feat_list:
                feat_val_list = list(cluster_centroid_dict[feat].keys())
                for feat_val_idx in range(len(feat_val_list)):
                    feat_val = feat_val_list[feat_val_idx]
                    centroid_list = cluster_centroid_dict[feat][feat_val]
                    fig, ax = plt.subplots(nrows=1, ncols=len(centroid_list))
                    instance_list = cluster_instance_dict[feat][feat_val]
                    for idx in range(len(centroid_list)):
                        cf = eval_obj.cf_df[(eval_obj.cf_df['feature'] == feat) & (eval_obj.cf_df['feat_value'] == feat_val) & (eval_obj.cf_df['centroid_idx'] == idx)]['normal_cf'].values[0]
                        bin_enc, cat_enc, bin_enc_cols, cat_enc_cols, binary, categorical, numerical = eval_obj.bin_enc, eval_obj.cat_enc, eval_obj.bin_enc_cols, eval_obj.cat_enc_cols, eval_obj.binary, eval_obj.categorical, eval_obj.numerical
                        centroid_original = inverse_transform_only(bin_enc, cat_enc, bin_enc_cols, cat_enc_cols, binary, categorical, numerical, centroid_list[idx]).values[0]
                        cf_df = pd.DataFrame(cf.reshape(1,-1), index = [0], columns=eval_obj.data_cols)
                        cf_original = inverse_transform_only(bin_enc, cat_enc, bin_enc_cols, cat_enc_cols, binary, categorical, numerical, cf_df).values[0]
                        if len(centroid_list) > 1:
                            ax[idx].plot(features, centroid_original, 'r--')
                            ax[idx].plot(features, cf_original, 'b--')
                            ax[idx].set_xlabel('Features')
                            # ax[idx].set_ylabel('Values')
                            # ax[idx].set_title(f'{method_str.upper()}, Cluster {idx+1}')
                            ax[idx].get_xaxis().set_visible(False)
                            ax[idx].set_xticklabels(features, rotation=45, ha='right')
                        else:
                            ax.plot(features, centroid_original, 'r--')
                            ax.plot(features, cf_original, 'b--')
                            ax.set_xlabel('Features')
                            # ax.set_ylabel('Values')
                            # ax.set_title(f'{method_str.upper()}, Cluster {idx+1}')
                            ax.get_xaxis().set_visible(False)
                            ax.set_xticklabels(features, rotation=45, ha='right')
                        # legend_list = [Line2D([0], [0], color=colors_list[feat_val_idx_plot], lw=3) for feat_val_idx_plot in range(len(feat_val_list))]
                        # if len(centroid_list) > 1:
                        #     ax[method_idx, idx].legend(legend_list, feat_val_list)
                        # else:
                        #     ax[method_idx].legend(legend_list, feat_val_list)
                    fig.suptitle(f'{method_str.upper()}')
                    # fig.tight_layout()
                    plt.savefig(f'{results_cf_plots_dir}{data_str}_{method_str}_{feat}_{feat_val}_centroids_cfs.pdf', format='pdf')
        

colors_list = ['red', 'blue', 'green', 'purple', 'lightgreen', 'tab:brown', 'orange']
colors_dict = {'All':'black','Male':'red','Female':'blue','White':'gainsboro','Non-white':'dimgray',
               '<25':'thistle','25-60':'violet','>60':'purple','<18':'green','>=18':'yellow','Single':'peachpuff',
               'Married':'peru','Divorced':'saddlebrown','isMarried: True':'cyan','isMarried: False':'darkcyan',
               'isMale: True':'lightcoral','isMale: False':'firebrick','Other':'lightgreen','HS':'limegreen',
               'University':'green','Graduate':'darkgreen','African-American':'orangered','Caucasian':'coral'}

# method_box_plot(datasets, methods_to_run, 'proximity', colors_list)
# fnr_plot(datasets, colors_dict)
# burden_plot(datasets, methods_to_run, colors_dict)
# fnr_burden_plot(datasets, methods_to_run, 'proximity', colors_list)
# nawb_plot(datasets, methods_to_run, colors_dict)
# validity_groups_cf(datasets, methods_to_run)
# validity_clusters(datasets, methods_to_run)
# burden_groups_cf(datasets, methods_to_run)
# burden_cluster_cf(datasets, methods_to_run)
# nawb_groups_cf(datasets, methods_to_run)
# nawb_cluster_cf(datasets, methods_to_run)
# burden_groups_cf_bar(datasets, 'NN')
# plot_centroids()
plot_centroids_cfs()


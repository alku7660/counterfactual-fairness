import numpy as np
import pandas as pd
from itertools import product
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, tuplelist
from collections import Counter
from data_constructor import load_dataset
from model_constructor import Model
from evaluator_constructor import distance_calculation, verify_feasibility
from nnt import nn_for_juice
import time
from scipy.stats import norm
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
import time
import copy

"""
This method is based on:
Kavouras, L., Tsopelas, K., Giannopoulos, G., Sacharidis, D., Psaroudaki, E., Theologitis, N., ... & Emiris, I. (2023). Fairness Aware Counterfactuals for Subgroups. arXiv preprint arXiv:2306.14978.
"""

class FACTS:

    def __init__(self, counterfactual) -> None:
        data = counterfactual.data
        model = counterfactual.model
        self.cluster = counterfactual.cluster
        self.model = model
        self.discretized_train_df = data.discretized_train_df
        self.train_target = data.train_target
        self.undesired_discretized_train_df, self.desired_discretized_train_df = self.get_undesired_desired_discretized_train_df
        self.discretized_test_df = data.discretized_test_df
        self.transformed_test_df = data.transformed_test_df
        self.transformed_test_np = data.transformed_test_np
        self.test_target = data.test_target
        self.undesired_class = data.undesired_class
        self.protected_groups = data.feat_protected
        start_time = time.time()
        self.sensitive_groups = self.get_sensitive_groups()
        self.fpgrowth_df = self.get_fpgrowth_df()
        self.fpgrowth_per_feat = self.get_common_fpgrowth_per_sensitive_feature()
        self.actions_fpgrowth_set = self.get_actions_fpgrowth_set()
        self.subgroup_same_cost_actions = self.get_same_cost_actions_per_subgroup(data)
        self.effectiveness_df = self.estimate_effectiveness_per_action_per_sensitive_group(self, data, model)
        self.best_effectiveness_df = self.select_best_action_per_subgroup()
        self.normal_x_cf = self.get_cfs_all_fn_instances(data)
        end_time = time.time()
        self.run_time = end_time - start_time

    def get_undesired_desired_discretized_train_df(self):
        """
        Obtains the undesired class and desired class training dataset
        """
        return self.discretized_train_df[self.train_target == self.undesired_class], self.discretized_train_df[self.train_target == 1 - self.undesired_class]

    def get_fpgrowth_df(self):
        """
        Obtains the fpgrowth conjunction predicates from the frequent itemsets from the fpgrowth algorithm, as explained in:
        Kavouras, L., Tsopelas, K., Giannopoulos, G., Sacharidis, D., Psaroudaki, E., Theologitis, N., ... & Emiris, I. (2023). Fairness Aware Counterfactuals for Subgroups. arXiv preprint arXiv:2306.14978.
        """
        frequent_subgroups_per_sensitive_group = {}
        for sensitive_feat in self.protected_groups.keys():
            sensitive_groups_dict = self.protected_groups[sensitive_feat]
            sensitive_feat_groups = {}
            for sensitive_group in sensitive_groups_dict.keys():
                column = f'{sensitive_feat}_{int(sensitive_group)}'
                instances_sensitive_group = self.undesired_discretized_train_df[self.undesired_discretized_train_df[column] == 1]
                instances_sensitive_group = instances_sensitive_group.drop(column)
                fpgrowth_sensitive_group_df = fpgrowth(instances_sensitive_group, min_support=0.01, use_colnames=True)
                sensitive_feat_groups[sensitive_group] = fpgrowth_sensitive_group_df
            frequent_subgroups_per_sensitive_group[sensitive_feat] = sensitive_feat_groups
        return frequent_subgroups_per_sensitive_group

    def get_common_fpgrowth_per_sensitive_feature(self):
        """
        Obtains the sensitive groups that are common among the sensitive groups for each of the sensitive features found in the dataset
        """
        common_frequent_subgroups_per_sensitive_feat = {}
        for sensitive_feat in self.protected_groups.keys():
            sensitive_subgroups_dict = self.fpgrowth_df[sensitive_feat]
            all_frequent_itemsets_sensitive_feat = pd.DataFrame()
            for sensitive_group in sensitive_subgroups_dict.keys():
                fp_growth_sensitive_group_df = sensitive_subgroups_dict[sensitive_group]
                all_frequent_itemsets_sensitive_feat = pd.concat((all_frequent_itemsets_sensitive_feat, fp_growth_sensitive_group_df))
            value_counts_sensitive_feature_subgroups = all_frequent_itemsets_sensitive_feat['itemsets'].value_counts()
            common_sensitive_groups_feat = value_counts_sensitive_feature_subgroups[value_counts_sensitive_feature_subgroups.iloc[:,-1] == len(sensitive_subgroups_dict.keys())]['itemsets']
            common_frequent_subgroups_per_sensitive_feat[sensitive_feat] = common_sensitive_groups_feat
        return common_frequent_subgroups_per_sensitive_feat

    def remove_sensitive_feature_itemsets(self, itemsets):
        """
        Removes the sensitive feature from the itemsets series
        """
        itemsets = list(itemsets)
        itemsets = [list(itemset) for itemset in itemsets]
        sensitive_feat_list = list(self.fpgrowth_per_feat.keys())
        for itemset in itemsets:
            for feat in itemset:
                if any(sensitive_feat in feat for sensitive_feat in sensitive_feat_list):
                    itemset.remove(feat)
        itemsets = pd.Series(itemsets)
        return itemsets

    def get_actions_fpgrowth_set(self):
        """
        Obtains actions from the unaffected training set
        """
        filtered_fpgrowth_actions_df = pd.DataFrame()
        fpgrowth_actions_df = fpgrowth(self.desired_discretized_train_df, min_support=0.01, use_colnames=True)
        fpgrowth_actions_srs = self.remove_sensitive_feature_itemsets(fpgrowth_actions_df['itemsets'])
        for sensitive_feat in self.fpgrowth_per_feat.keys():
            common_frequent_subgroups_per_sensitive_feat_df = self.fpgrowth_per_feat[sensitive_feat]
            fpgrowth_actions_to_subgroups = common_frequent_subgroups_per_sensitive_feat_df.loc[common_frequent_subgroups_per_sensitive_feat_df.isin(fpgrowth_actions_srs)]
            filtered_fpgrowth_actions_df =pd.concat((filtered_fpgrowth_actions_df, fpgrowth_actions_to_subgroups))
        return filtered_fpgrowth_actions_df
    
    def get_instances_idx_belonging_to_subgroup(self, subgroup):
        """
        Obtains the instances indices in the false negatives belonging to a given subgroup
        """
        subgroup_instances_idx = []
        for c_idx in range(1, len(self.cluster.filtered_clusters_list) + 1):
            cluster_instances_list = self.cluster.filtered_clusters_list[c_idx - 1]
            for instance_idx in cluster_instances_list:
                instance = self.discretized_test_df.loc[instance_idx]
                instance_feat_values = [int(instance[feat].values) for feat in subgroup]
                if instance_feat_values == [1]*len(subgroup):
                    subgroup_instances_idx.append(instance_idx)
        return subgroup_instances_idx
    
    def get_instances_with_idx(self, idx_list):
        """
        Obtains the instances in discretized form according to a list of indices
        """
        instances_group = self.discretized_test_df.loc[idx_list]
        return instances_group
    
    def get_all_columns_with_feat(self, feat):
        """
        Get all the columns with the feature given as input parameter
        """
        columns_list = [col for col in self.discretized_test_df.columns if feat in col]
        return columns_list

    def get_x_discretized_prime_from_discretized_x(self, action, x):
        """
        Transforms x to x_prime using the action given
        """
        x_prime = copy.deepcopy(x)
        action_feat = [action.split('_')[0]] if len_action == 1 else [i.split('_') for i in action]
        len_action = 1 if isinstance(action, str) else len(action)
        columns_with_feat = self.get_all_columns_with_feat(action_feat)
        x_prime[columns_with_feat] = [0]*len(columns_with_feat)
        x_prime[action] = [1]*len_action
        return x_prime

    def transform_to_normal_x(self, x, data):
        """
        Transforms an instance x from discretized form to normal x form 
        """
        x_original = data.decode_df(x)
        x_transformed = data.transform_data(x_original)
        return x_transformed

    def adjust_continuous_feat_normal_x_prime(self, x_prime, x_transformed, action, data):
        """
        Adjusts x_prime continuous features to normal values if not changed by the action given
        """
        x_prime_transformed = self.transform_to_normal_x(x_prime, data)
        len_action = len(action)
        for cont_feat in data.continuous:
            action_feat = [action.split('_')[0]] if len_action == 1 else [i.split('_') for i in action]
            if cont_feat not in action_feat:
                x_prime_transformed[cont_feat] = x_transformed[cont_feat].values
        return x_prime_transformed

    def verify_same_cost_subgroup_action(self, subgroup_instances, action, data):
        """
        Verifies the same cost for all instances in a subgroup for the given action
        """
        cost_instances = []
        for x in subgroup_instances:
            x_transformed = data.transformed_test_df.loc[x.index,:]
            x_prime = self.get_x_discretized_prime_from_discretized_x(action, x)
            x_prime_normal = self.transform_to_normal_x(x_prime, data)
            cost = distance_calculation(np.array(x_transformed), np.array(x_prime_normal), data, type='L1_L0')
            cost_instances.append(cost)
        return len(set(cost_instances)) == 1

    def get_same_cost_actions_per_subgroup(self, data):
        """
        Obtains the subset of actions that have the same cost for all individuals in each subgroup, and that mention at least 1 feature mentioned and at least 1 different feature value.
        """
        action_per_subgroup_dict = {}
        for sensitive_feat in self.protected_groups.keys():
            subgroups = list(self.fpgrowth_per_feat[sensitive_feat])
            for subgroup in subgroups:
                subgroup_feat = [subgroup.split('_')[0]] if len(subgroup) == 1 else [i.split('_') for i in subgroup]
                subgroup_instances_idx = self.get_instances_idx_belonging_to_subgroup(subgroup)
                subgroup_instances = self.get_instances_with_idx(subgroup_instances_idx)
                same_cost_actions_list = []
                for action in self.actions_fpgrowth_set:
                    action_feat = [action.split('_')[0]] if len(action) == 1 else [i.split('_') for i in action]
                    if subgroup.sort() != action.sort() and any(x in subgroup_feat for x in action_feat): 
                        if self.verify_same_cost_subgroup_action(subgroup_instances, action, data):
                            same_cost_actions_list.append(action)
                    action_per_subgroup_dict[subgroup] = same_cost_actions_list
        return action_per_subgroup_dict

    def calculate_action_effectiveness(self, subgroup, sensitive_group, action, data, model):
        """
        Given a subgroup, sensitive group, and an action, calculates the effectiveness according to Kavouras, L., Tsopelas, K., Giannopoulos, G., Sacharidis, D., Psaroudaki, E., Theologitis, N., ... & Emiris, I. (2023). Fairness Aware Counterfactuals for Subgroups. arXiv preprint arXiv:2306.14978.
        """
        subgroup_instances_idx = self.get_instances_idx_belonging_to_subgroup(subgroup)
        subgroup_instances_df = self.get_instances_with_idx(subgroup_instances_idx)
        subgroup_instances_sensitive_group_df = subgroup_instances_df.loc[subgroup_instances_df[sensitive_group] == 1]
        effective = 0
        for x in subgroup_instances_sensitive_group_df:
            normal_x = self.transformed_test_df.loc[x.index,:]
            x_prime = self.get_x_discretized_prime_from_discretized_x(action, x)
            x_prime_normal = self.transform_to_normal_x(x_prime, data)
            x_prime_normal = self.adjust_continuous_feat_normal_x_prime(x_prime_normal, normal_x, action, data)
            x_pred = model.model.predict(normal_x.values)
            x_prime_pred = model.model.predict(x_prime_normal.values)
            if x_pred == data.undesired_class and x_prime_pred != data.undesired_class:
                effective += 1
            else:
                effective += 0
        effectiveness = effective/len(subgroup_instances_sensitive_group_df)
        return effectiveness

    def estimate_effectiveness_per_action_per_sensitive_group(self, data, model):
        """
        Estimates the effectiveness of each of the actions for each of the subgroups they apply to. Effectiveness here is simply whether they change the label or not (feasibility is not considered)
        """
        cols = ['subgroup','sensitive_group','action','effectiveness']
        effectiveness_df = pd.DataFrame(columns=cols)
        count = 0
        for subgroup in self.subgroup_same_cost_actions.keys():
            actions_list = self.subgroup_same_cost_actions[subgroup]
            for sensitive_group in self.sensitive_groups:
                for action in actions_list:
                    count += 1
                    effectiveness = self.calculate_action_effectiveness(subgroup, sensitive_group, action, data, model)
                    effectiveness_row = pd.DataFrame(data=[subgroup, sensitive_group, action, effectiveness], index = [count], columns=cols)
                    effectiveness_df = pd.concat((effectiveness_df, effectiveness_row))
        effectiveness_df.sort_values('effectiveness', ascending=False)
        return effectiveness_df

    def select_best_action_per_subgroup(self):
        """
        Selects the best action for every subgroup based on effectiveness (maximum effectiveness per subgroup)
        """
        cols = ['subgroup','sensitive_group','action','effectiveness']
        best_effectiveness_action_df = pd.DataFrame(columns=cols)
        unique_subgroups = self.effectiveness_df['subgroup'].unique()
        for unique_subgroup in unique_subgroups:
            subgroup_effectiveness_df = self.effectiveness_df[self.effectiveness_df['subgroup'] == unique_subgroup]
            best_effectiveness_for_subgroup = subgroup_effectiveness_df.iloc[0,:]
            best_effectiveness_action_df = pd.concat((best_effectiveness_action_df, best_effectiveness_for_subgroup))
        return best_effectiveness_action_df

    def get_cfs_all_fn_instances(self, data):
        """
        Gets the counterfactuals, based on the best actions, for the given subgroups and for each instance
        """
        cfs_dict = dict()
        for row in self.best_effectiveness_df:
            subgroup, action = row['subgroup'], row['action']
            subgroup_instances_idx = self.get_instances_idx_belonging_to_subgroup(subgroup)
            subgroup_instances_df = self.get_instances_with_idx(subgroup_instances_idx)
            for subgroup_instance in subgroup_instances_df:
                idx = subgroup_instance.index
                x_transformed = self.transformed_test_df.loc[subgroup_instance.index,:]
                x_discretized_prime = self.get_x_discretized_prime_from_discretized_x(action, subgroup_instance)
                x_prime_normal = self.transform_to_normal_x(x_discretized_prime, data)
                x_prime_normal = self.adjust_continuous_feat_normal_x_prime(x_prime_normal, x_transformed, action, data)
                cfs_dict[idx] = x_prime_normal
        return cfs_dict

    def get_sensitive_groups(self):
        """
        Obtains a list of sensitive groups
        """
        sensitive_group_list = []
        for sensitive_group in self.protected_groups.keys():
            sensitive_group_list.extend([x for x in self.discretized_train_df.columns if sensitive_group in x])
        return sensitive_group_list
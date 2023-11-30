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
        self.recourse_predicates_per_group = self.get_recourse_predicates_per_sensitive_group()
        self.fn_instances = self.get_fn_instances()
        self.coverage_dict = self.preallocate_all_group_predicate_R()
        self.results_df = pd.DataFrame(columns=['x_idx', 'q', 'q_c', 'q_c_c_prime', 'correctness', 'feat_change'])
        self.best_recourse_df = pd.DataFrame(columns=['x_idx', 'best_q_c_c_prime', 'correctness', 'feat_change'])
        self.get_recourses_for_fn_instances(data, model)
        self.get_recourses_for_centroids(data, model)
        self.format_results()
        self.get_best_recourse_rule_x()
        self.unique_recourse_rules_df = self.get_unique_recourse_rules()
        self.normal_x_cf = self.get_cf_normal_x_form(data)
        self.normal_x_cf_centroids = self.get_cf_normal_x_form_centroids(data)
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
    
    def get_x_discretized_prime_from_discretized_x(self, action, x):
        """
        Transforms x to x_prime using the action given
        """
        x_prime = copy.deepcopy(x)
        len_action = 1 if isinstance(action, str) else len(action)
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
    
    x_pred = model.model.predict(x_transformed.values)
    x_prime_pred = model.model.predict(x_prime_transformed.values)
    if x_pred == data.undesired_class and x_prime_pred != data.undesired_class:
        correctness_q_c_c_prime = 1
    else:
        correctness_q_c_c_prime = 0

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
                for action in self.actions_fpgrowth_set:
                    action_feat = [action.split('_')[0]] if len(action) == 1 else [i.split('_') for i in action]
                    if subgroup.sort() != action.sort() and any(x in subgroup_feat for x in action_feat): 
                        if self.verify_same_cost_subgroup_action(subgroup_instances, action, data):
                            action_per_subgroup_dict[subgroup] = action
        return action_per_subgroup_dict

    def estimate_effectiveness_per_action_per_sensitive_group(self):
        """
        Estimates the effectiveness of each of the actions towards 
        """
        eta = {}
        len_cluster_instances = 0
        for c_idx in range(1, len(self.cluster.filtered_clusters_list) + 1):
            cluster_instances_list = self.cluster.filtered_clusters_list[c_idx - 1]
            len_cluster_instances += len(cluster_instances_list)
        for cf_key in list(cfs.keys()):
            sum_eta = 0
            cf_k = cfs[cf_key]
            cf_k = cf_k.values[0]
            for c_idx in range(1, len(self.cluster.filtered_clusters_list) + 1):
                cluster_instances_list = self.cluster.filtered_clusters_list[c_idx - 1]
                for instance_idx in cluster_instances_list:
                    instance = self.cluster.transformed_false_undesired_test_df.loc[instance_idx].values
                    sum_eta += verify_feasibility(instance, cf_k, data)
            eta[cf_key] = sum_eta/len_cluster_instances
            print(f'Highest eta value: {np.max(list(eta.values()))}')
        return eta

    def get_sensitive_groups(self):
        """
        Obtains a list of sensitive groups
        """
        sensitive_group_list = []
        for sensitive_group in self.protected_groups.keys():
            sensitive_group_list.extend([x for x in self.discretized_train_df.columns if sensitive_group in x])
        return sensitive_group_list 

    def get_recourse_predicates_per_sensitive_group(self):
        """
        Obtains a Dict object containing the frequent itemsets (recourse predicates) for each of the sensitive groups.
        """
        predicate_dict = {}
        for sensitive_group in self.sensitive_groups:
            itemset_list = [list(itemset) for itemset in self.apriori_df.itemsets.values if sensitive_group in itemset]
            for itemset in itemset_list:
                itemset.remove(sensitive_group) 
            itemset_list = [itemset for itemset in itemset_list if len(itemset) > 0]
            predicate_dict[sensitive_group] = itemset_list
        return predicate_dict
    
    def get_fn_instances(self):
        """
        Obtains the set of instances that belong to the false negative class in the test set.
        """
        prediction_label_df = pd.DataFrame(index=self.transformed_test_df.index, data=np.array([self.model.model.predict(self.transformed_test_np), self.test_target]).T, columns=['prediction','label'])
        false_negatives_label_df = prediction_label_df.loc[(prediction_label_df['prediction'] == self.undesired_class) & (prediction_label_df['label'] != self.undesired_class)]
        false_negatives_idx = false_negatives_label_df.index
        false_negatives_instances = self.discretized_test_df.loc[false_negatives_idx,:]
        return false_negatives_instances
    
    def find_sensitive_group(self, x):
        """
        Obtains the sensitive groups an instance x belongs to as a list
        """
        q_i_list = []
        for sensitive_group in self.sensitive_groups:
            if x[sensitive_group] == 1:
                q_i = sensitive_group
                q_i_list.append(q_i)
        return q_i_list
    
    def preallocate_all_group_predicate_R(self):
        """
        Preallocates the coverage for all the pairs (q_i, c_i)
        """
        all_groups_predicates = []
        coverage_dict = dict()
        for key, val in self.recourse_predicates_per_group.items():
            for val_i in val:
                all_groups_predicates.append(tuple([key] + val_i))
        for key_value in all_groups_predicates:
            coverage_dict[key_value] = 0
        return coverage_dict

    def find_sensitive_group_x(self, x):
        """
        Obtains the sensitive groups x belongs to
        """
        q_list = []
        for q in self.sensitive_groups:
            if int(x[q].values) == 1:
                q_list.append(q)
        return q_list

    def find_recourse_predicate_x_q(self, x, q_list):
        """
        Obtains the set of C predicates corresponding to the given x and q group and sums up coverage for C
        """
        c_i_dict = {}
        for q in q_list:
            recourse_predicates_q = self.recourse_predicates_per_group[q]
            c_i_list = []
            for c in recourse_predicates_q:
                c_values = [int(x[c_feat].values) for c_feat in c]
                if c_values == [1]*len(c):
                    c_i_list.append(c)
                    c_key = tuple([q] + c)
                    if c_key in self.coverage_dict.keys():
                        self.coverage_dict[c_key] += 1
            c_i_dict[q] = c_i_list
        return c_i_dict
    
    def find_recourse_rules(self, q, c):
        """
        Given a Q and a C, obtains the C's available in the frequent itemsets belonging to Q
        """
        c_prime_list = copy.deepcopy(self.recourse_predicates_per_group[q])
        c_list_copy = copy.deepcopy(c_prime_list)
        for features in c:
            feat_name, _ = features.split('_')
            for c_i in c_list_copy:
                c_i_name_list = [i.split('_')[0] for i in c_i]
                if feat_name not in c_i_name_list or len(c) != len(c_i_name_list):
                    if c_i in c_prime_list:
                        c_prime_list.remove(c_i)
            if c in c_prime_list:
                c_prime_list.remove(c)
        return c_prime_list
                    
    def get_all_recourse_rules_x(self, sensitive_groups_x, recourse_predicates_x):
        """
        Obtains all recourse rules for instance x
        """
        q_to_c_dict = dict()
        for q in sensitive_groups_x:
            c_to_c_prime_dict = dict()
            for c in recourse_predicates_x[q]:
                c_prime = self.find_recourse_rules(q, c)
                c_key = tuple(c)[0] if len(c) == 1 else tuple(c)
                c_to_c_prime_dict[c_key] = c_prime
            q_to_c_dict[q] = c_to_c_prime_dict
        return q_to_c_dict
    
    def extract_recourses_x(self, x):
        """
        Gets the recourse set for an instance x
        """
        sensitive_groups_x = self.find_sensitive_group_x(x)
        recourse_predicates_x = self.find_recourse_predicate_x_q(x, sensitive_groups_x)
        recourse_rules = self.get_all_recourse_rules_x(sensitive_groups_x, recourse_predicates_x)
        return recourse_rules
    
    def preallocate_correctness_feat_change_x(self, recourse_rules_x):
        """
        Preallocates the dictionary for correctness and feature change
        """
        correctness_dict_x = dict()
        feat_change_dict_x = dict()
        for q in recourse_rules_x.keys():
            c_dict = recourse_rules_x[q]
            c_prime_dict_corr = dict()
            c_prime_dict_feat = dict()
            for c in c_dict.keys():
                c_prime_list = c_dict[c]
                correct = dict()
                feat_change = dict()
                for c_prime in c_prime_list:
                    c_prime_key = tuple(c_prime)[0] if len(c_prime) == 1 else tuple(c_prime)
                    correct[c_prime_key] = 0
                    feat_change[c_prime_key] = 0
                c_prime_dict_corr[c] = correct
                c_prime_dict_feat[c] = feat_change
            correctness_dict_x[q] = c_prime_dict_corr
            feat_change_dict_x[q] = c_prime_dict_feat
        return correctness_dict_x, feat_change_dict_x

    def preallocate_all_instances(self, data):
        """
        Preallocate all instances into the counters structure
        """
        correctness_dict, feat_change_dict = Counter(), Counter()
        for x_fn_i in range(len(self.fn_instances.index)):
            x_fn_idx = self.fn_instances.index[x_fn_i]
            x = data.discretized_test_df.loc[x_fn_idx,:].to_frame().T
            recourse_set = self.extract_recourses_x(x)
            correctness_dict_x, feat_change_dict_x = self.preallocate_correctness_feat_change_x(recourse_set)
            if x_fn_i == 0:
                correctness_dict.update(correctness_dict_x)
                feat_change_dict.update(feat_change_dict_x)
            else:
                for q in correctness_dict_x.keys():
                    if q not in correctness_dict.keys():
                        correctness_dict[q] = correctness_dict_x[q]
                        feat_change_dict[q] = feat_change_dict_x[q]
                    else:
                        for c in correctness_dict_x[q].keys():
                            if c not in correctness_dict[q].keys():
                                correctness_dict[q][c] = correctness_dict_x[q][c]
                                feat_change_dict[q][c] = feat_change_dict_x[q][c]
                            else:
                                for c_prime in correctness_dict_x[q][c].keys():
                                    if c_prime not in correctness_dict[q][c].keys():
                                        correctness_dict[q][c][c_prime] = correctness_dict_x[q][c][c_prime]
                                        feat_change_dict[q][c][c_prime] = feat_change_dict_x[q][c][c_prime]
                                    else:
                                        continue
        return correctness_dict, feat_change_dict

    def results_recourse_rules_x(self, recourse_rules_x, x, data, model):
        """
        Adds the recourse rules obtained for x, their correctness to the correctness dictionary and the feature change to the change dictionary
        """
        if not isinstance(x.index[0], str):
            x_idx = int(x.index[0])
            x_transformed = data.transformed_test_df.loc[x.index,:]
        else:
            x_idx = x.index[0]
            x_idx_c_idx = int(x_idx.split('_')[-1])
            x_transformed = self.cluster.filtered_centroids_list[x_idx_c_idx].normal_x_df
        results_x_list = []
        for q in recourse_rules_x.keys():
            c_dict = recourse_rules_x[q]
            for c in c_dict.keys():
                c_prime_list = c_dict[c]
                c_key = c if isinstance(c, str) else tuple(c)
                c_list = c if isinstance(c, str) else list(c)
                q_c = (q, c_key)
                len_c = 1 if isinstance(c, str) else len(c)
                for c_prime in c_prime_list:
                    x_prime = copy.deepcopy(x)
                    x_prime[c_list] = [0]*len_c
                    c_prime_key = tuple(c_prime)[0] if len(c_prime) == 1 else tuple(c_prime)
                    q_c_c_prime = (q, c_key, c_prime_key)
                    len_c_prime_key = 1 if isinstance(c_prime_key, str) else len(c_prime_key)
                    x_prime[c_prime] = np.array([1]*len_c_prime_key)
                    x_prime_transformed = self.transform_to_normal_x(x_prime, data)
                    c_prime_key_name_list = [c_prime_key.split('_')[0]] if len_c_prime_key == 1 else [i.split('_') for i in c_prime_key]
                    for cont_feat in data.continuous:
                        if cont_feat not in c_prime_key_name_list:
                            x_prime_transformed[cont_feat] = x_transformed[cont_feat].values
                    x_pred = model.model.predict(x_transformed.values)
                    x_prime_pred = model.model.predict(x_prime_transformed.values)
                    if x_pred == data.undesired_class and x_prime_pred != data.undesired_class:
                        correctness_q_c_c_prime = 1
                    else:
                        correctness_q_c_c_prime = 0
                    feat_change_q_c_c_prime = distance_calculation(np.array(x_transformed), np.array(x_prime_transformed), data, type='L1_L0')
                    result_x = [x_idx, q, q_c, q_c_c_prime, correctness_q_c_c_prime, feat_change_q_c_c_prime]
                    results_x_list.append(result_x)
        return results_x_list
    
    def change_x_to_x_prime(self, x, q_c_c_prime):
        """
        Transforms and instance x to x_prime by using the given recourse rule. x must be in discretized form.
        """
        x_prime = copy.deepcopy(x)
        c, c_prime = q_c_c_prime[1], q_c_c_prime[2]
        c = c if isinstance(c,str) else list(c)
        c_prime = c_prime if isinstance(c_prime,str) else list(c_prime)
        len_c = 1 if isinstance(c, str) else len(c)
        x_prime[c] = [0]*len_c
        len_c_prime_key = 1 if isinstance(c_prime, str) else len(c_prime)
        x_prime[c_prime] = np.array([1]*len_c_prime_key)
        return x_prime

    def add_results(self, results_x):
        """
        Joins the list of results of x holding all the results with new instance dictionaries from ARES to the DataFrame containing everything
        """
        results_x_df = pd.DataFrame(data=results_x, columns=self.results_df.columns)
        self.results_df = pd.concat((self.results_df, results_x_df))
    
    def get_recourses_for_fn_instances(self, data, model):
        """
        Obtains all the best recourses for all FN instances
        """
        counter = 1
        set_instances = self.fn_instances.index
        for x_fn_idx in set_instances:
            start_time = time.time()
            x = data.discretized_test_df.loc[x_fn_idx,:].to_frame().T
            recourse_set = self.extract_recourses_x(x)
            results_x = self.results_recourse_rules_x(recourse_set, x, data, model)
            self.add_results(results_x)
            end_time = time.time()
            print(f'Dataset: {data.name}. Instance {x_fn_idx} ({counter}/{len(set_instances)}) done (time: {np.round(end_time - start_time, 2)} s)')
            counter += 1
    
    def get_recourses_for_centroids(self, data, model):
        """
        Obtains the best recourses for centroids
        """
        counter = 1
        for c_idx in range(len(self.cluster.filtered_centroids_list)):
            start_time = time.time()
            centroid = self.cluster.filtered_centroids_list[c_idx]
            original_centroid = pd.DataFrame(data=centroid.x.reshape(1,-1), index=[f'c_{c_idx}'], columns=data.features)
            discretized_centroid = data.discretize_df(original_centroid)
            recourse_set = self.extract_recourses_x(discretized_centroid)
            results_centroid = self.results_recourse_rules_x(recourse_set, discretized_centroid, data, model)
            self.add_results(results_centroid)
            end_time = time.time()
            print(f'Dataset: {data.name}. Centroid {c_idx} ({counter}/{len(self.cluster.filtered_centroids_list)}) done (time: {np.round(end_time - start_time, 2)} s)')
            counter += 1

    def format_results(self):
        """
        Changes the result dictionaries into a more readable form (this is used for correctness and feature change dictionaries)
        """
        total_instances_q_c_c_prime_df = self.results_df['q_c_c_prime'].value_counts()
        correct_instances_q_c_c_prime_df = self.results_df[self.results_df['correctness'] == 1]['q_c_c_prime'].value_counts()
        unique_q_c_c_prime = list(total_instances_q_c_c_prime_df.index)
        q_c_c_prime_list = []
        for q_c_c_prime in unique_q_c_c_prime:
            total_instances_q_c_c_prime = total_instances_q_c_c_prime_df[q_c_c_prime]
            try:
                correct_instances_q_c_c_prime = correct_instances_q_c_c_prime_df[q_c_c_prime]
            except:
                correct_instances_q_c_c_prime = 0
            correct_over_total_q_c_c_prime = correct_instances_q_c_c_prime/total_instances_q_c_c_prime
            q_c_c_prime_list.append([q_c_c_prime, total_instances_q_c_c_prime, correct_instances_q_c_c_prime, correct_over_total_q_c_c_prime])
        self.correctness_df = pd.DataFrame(data=q_c_c_prime_list, columns=['q_c_c_prime','total_instances','correct_instances','correct_over_total']).sort_values(['correct_over_total','correct_instances'], ascending=[False, False])

    def get_best_recourse_rule_x(self):
        """
        Selects the best recourse rule for every instance x in the set of false negatives
        """
        for x_idx in self.results_df['x_idx'].unique():
            filter_x_correct_result_df = self.results_df[(self.results_df['x_idx'] == x_idx) & (self.results_df['correctness'] == 1)]
            for best_q_c_c_prime in list(self.correctness_df['q_c_c_prime']):
                if best_q_c_c_prime in list(filter_x_correct_result_df['q_c_c_prime']):
                    found_best_q_c_c_prime = best_q_c_c_prime
                    feat_change_best_q_c_c_prime = filter_x_correct_result_df[filter_x_correct_result_df['q_c_c_prime'] == found_best_q_c_c_prime]['feat_change'].values[0]
                    break
            best_recourse_x = [x_idx, found_best_q_c_c_prime, 1, feat_change_best_q_c_c_prime]
            best_recourse_x_df = pd.DataFrame(data=[best_recourse_x], columns=self.best_recourse_df.columns)
            self.best_recourse_df = pd.concat((self.best_recourse_df, best_recourse_x_df))

    def get_unique_recourse_rules(self):
        """
        Obtains the unique rules used
        """
        unique_recourse_rules_df = self.best_recourse_df['best_q_c_c_prime'].value_counts()
        return unique_recourse_rules_df

    def get_cf_normal_x_form_centroids(self, data):
        """
        Obtains the normal x form of the CF obtained through the recourse rules for the centroids
        """
        normal_x_cf_centroids = dict()
        for c_idx in range(len(self.cluster.filtered_centroids_list)):
            idx = f'c_{c_idx}'
            centroid = self.cluster.filtered_centroids_list[c_idx]
            original_centroid = pd.DataFrame(data=centroid.x.reshape(1,-1), index=[f'c_{c_idx}'], columns=data.features)
            x = data.discretize_df(original_centroid)
            q_c_c_prime = self.best_recourse_df[self.best_recourse_df['x_idx'] == idx]['best_q_c_c_prime'][0]
            c_prime = q_c_c_prime[-1]
            c_prime_key = tuple(c_prime)[0] if len(c_prime) == 1 else tuple(c_prime)
            len_c_prime_key = 1 if isinstance(c_prime_key, str) else len(c_prime_key)
            x_prime = self.change_x_to_x_prime(x, q_c_c_prime)
            x_prime_normal = self.transform_to_normal_x(x_prime, data)
            c_prime_key_name_list = [c_prime_key.split('_')[0]] if len_c_prime_key == 1 else [i.split('_') for i in c_prime_key]
            for cont_feat in data.continuous:
                if cont_feat not in c_prime_key_name_list:
                    x_prime_normal[cont_feat] = centroid.normal_x_df[cont_feat].values
            normal_x_cf_centroids[idx] = x_prime_normal
        return normal_x_cf_centroids

    def get_cf_normal_x_form(self, data):
        """
        Obtains the normal x form for all unique q_c_c_prime recourse rules for all the false negatives, including centroids
        """
        normal_x_cf = dict()
        for idx in list(self.best_recourse_df['x_idx']):
            q_c_c_prime_idx = self.best_recourse_df[self.best_recourse_df['x_idx'] == idx]['best_q_c_c_prime'][0]
            c_prime = [q_c_c_prime_idx[-1]]
            c_prime_key = tuple(c_prime)[0] if len(c_prime) == 1 else tuple(c_prime)
            len_c_prime_key = 1 if isinstance(c_prime_key, str) else len(c_prime_key)
            if isinstance(idx,str):
                continue
            else:
                x_idx = self.discretized_test_df.loc[idx].to_frame().T
            x_prime = self.change_x_to_x_prime(x_idx, q_c_c_prime_idx)
            x_prime_normal = self.transform_to_normal_x(x_prime, data)
            c_prime_key_name_list = [c_prime_key.split('_')[0]] if len_c_prime_key == 1 else [i.split('_') for i in c_prime_key]
            for cont_feat in data.continuous:
                if cont_feat not in c_prime_key_name_list:
                    x_prime_normal[cont_feat] = self.transformed_test_df.loc[idx][cont_feat]
            normal_x_cf[idx] = x_prime_normal
        return normal_x_cf
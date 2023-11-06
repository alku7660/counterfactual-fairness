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
import time
import copy

"""
This method is based on:
Rawal, Kaivalya, and Himabindu Lakkaraju. "Beyond individualized recourse: Interpretable and interactive summaries of actionable recourses." Advances in Neural Information Processing Systems 33 (2020): 12187-12198.
"""

class ARES:

    def __init__(self, counterfactual) -> None:
        data = counterfactual.data
        model = counterfactual.model
        self.cluster = counterfactual.cluster
        self.model = model
        self.discretized_train_df = data.discretized_train_df
        self.discretized_test_df = data.discretized_test_df
        self.transformed_test_df = data.transformed_test_df
        self.transformed_test_np = data.transformed_test_np
        self.test_target = data.test_target
        self.undesired_class = data.undesired_class
        self.protected_groups = data.feat_protected
        self.sensitive_groups = self.get_sensitive_groups()
        self.apriori_df = self.get_apriori_df()
        self.recourse_predicates_per_group = self.get_recourse_predicates_per_sensitive_group()
        self.fn_instances = self.get_fn_instances()
        self.coverage_dict = self.preallocate_all_group_predicate_R()
        self.results_df = pd.DataFrame(columns=['x_idx', 'q', 'q_c', 'q_c_c_prime', 'correctness', 'feat_change'])
        self.best_recourse_df = pd.DataFrame(columns=['x_idx', 'best_q_c_c_prime', 'correctness', 'feat_change'])
    
    def get_apriori_df(self):
        """
        Obtains the apriori conjunction predicates from the frequent itemsets from the apriori algorithm, as explained in:
        Rawal, Kaivalya, and Himabindu Lakkaraju. "Beyond individualized recourse: Interpretable and interactive summaries of actionable recourses." Advances in Neural Information Processing Systems 33 (2020): 12187-12198.
        """
        apriori_df = apriori(self.discretized_train_df, min_support=0.01, use_colnames=True)
        return apriori_df

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
            recourse_set = ares.extract_recourses_x(x)
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
        x_idx = int(x.index[0])
        x_transformed = data.transformed_test_df.loc[x.index,:]
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
                    x_prime_original = data.decode_df(x_prime)
                    x_prime_transformed = data.transform_data(x_prime_original)
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
    
    def add_results(self, results_x):
        """
        Joins the list of results of x holding all the results with new instance dictionaries from ARES to the DataFrame containing everything
        """
        results_x_df = pd.DataFrame(data=results_x, columns=self.results_df.columns)
        self.results_df = pd.concat((self.results_df, results_x_df))
    
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

data_str = 'synthetic_athlete'
train_fraction = 0.7
seed = 12345
step = 0.01
data = load_dataset(data_str, train_fraction, seed, step)
model = Model(data)
ares = ARES(data, model)
counter = 1
for x_fn_idx in ares.fn_instances.index[:10]:
    start_time = time.time()
    x = data.discretized_test_df.loc[x_fn_idx,:].to_frame().T
    recourse_set = ares.extract_recourses_x(x)
    results_x = ares.results_recourse_rules_x(recourse_set, x, data, model)
    ares.add_results(results_x)
    end_time = time.time()
    print(f'Dataset: {data_str}. Instance {x_fn_idx} ({counter}/{len(ares.fn_instances.index)}) done (time: {np.round(end_time - start_time, 2)} s)')
    counter += 1
counter = 1
for c_idx in range(len(ares.cluster.filtered_centroids_list)):
    start_time = time.time()
    centroid = ares.cluster.filtered_centroids_list[c_idx]
    original_centroid = pd.DataFrame(data=centroid.x.reshape(1,-1), index=[0], columns=data.features)
    discretized_centroid = data.discretize_df(original_centroid)
    recourse_set = ares.extract_recourses_x(discretized_centroid)
    results_centroid = ares.results_recourse_rules_x(recourse_set, discretized_centroid, data, model)
    ares.add_results(results_centroid)
    end_time = time.time()
    print(f'Dataset: {data_str}. Centroid {c_idx} ({counter}/{len(ares.cluster.filtered_centroids_list)}) done (time: {np.round(end_time - start_time, 2)} s)')
    counter += 1
ares.format_results()
ares.get_best_recourse_rule_x()
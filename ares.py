import numpy as np
import pandas as pd
from itertools import product
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, tuplelist
from data_constructor import load_dataset
from model_constructor import Model
from evaluator_constructor import distance_calculation, verify_feasibility
from nnt import nn_for_juice
import time
from scipy.stats import norm
from mlxtend.frequent_patterns import apriori
import copy

"""
This method is based on:
Rawal, Kaivalya, and Himabindu Lakkaraju. "Beyond individualized recourse: Interpretable and interactive summaries of actionable recourses." Advances in Neural Information Processing Systems 33 (2020): 12187-12198.
"""

class ARES:

    def __init__(self, data, model) -> None:
        self.model = model
        self.discretized_train_df = data.discretized_train_df
        self.discretized_test_df = data.discretized_test_df
        self.transformed_test_df = data.transformed_test_df
        self.test_target = data.test_target
        self.undesired_class = data.undesired_class
        self.protected_groups = data.feat_protected
        self.sensitive_groups = self.get_sensitive_groups()
        self.apriori_df = self.get_apriori_df()
        self.recourse_predicates_per_group = self.get_recourse_predicates_per_sensitive_group()
        self.fn_instances = self.get_fn_instances()
        self.coverage_dict = self.preallocate_all_group_predicate_R()
        self.correctness_dict = dict()
    
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
        prediction_label_df = pd.DataFrame(index=self.transformed_test_df.index, data=np.array([self.model.model.predict(self.transformed_test_df), self.test_target]).T, columns=['prediction','label'])
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
        Obtains the set of all pairs (q_i, c_i)
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
        c_prime_dict = dict()
        for q in sensitive_groups_x:
            for c in recourse_predicates_x[q]:
                c_prime = self.find_recourse_rules(q, c)
                c_prime_dict[tuple([q] + c)] = c_prime
        return c_prime_dict
    
    def extract_recourses_x(self, x):
        """
        Gets the recourse set for an instance x
        """
        sensitive_groups_x = self.find_sensitive_group_x(x)
        recourse_predicates_x = self.find_recourse_predicate_x_q(x, sensitive_groups_x)
        recourse_rules = self.get_all_recourse_rules_x(sensitive_groups_x, recourse_predicates_x)
        return recourse_rules
    
    def add_recourse_rules_x(self, recourse_rules_x):
        """
        Adds the recourse rules obtained for x and its correctness to the correctness dictionary
        """
        self.correctness_dict[]
        
data_str = 'synthetic_athlete'
train_fraction = 0.7
seed = 12345
step = 0.01
data = load_dataset(data_str, train_fraction, seed, step)
model = Model(data)
ares = ARES(data, model)
x = data.discretized_test_df.iloc[0,:].to_frame().T
x_idx = int(x.index)
recourse_set = ares.extract_recourses_x(x)

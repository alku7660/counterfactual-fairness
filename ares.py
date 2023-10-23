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
    
    
    
    def find_recourse_predicate_x_q(self, x, q_list):
        """
        Obtains the set of C predicates corresponding to the given x and q group
        """
        c_i_dict = {}
        for q in q_list:
            recourse_predicates_q = self.sensitive_groups[q]
            c_i_list = []
            for c in recourse_predicates_q:
                if list(int(x[c].values)) == [1]*len(c):
                    c_i_list.append(c)
            c_i_dict[q] = c_i_list
        return c_i_dict
    

    
        
data_str = 'synthetic_athlete'
train_fraction = 0.7
seed = 12345
step = 0.01
data = load_dataset(data_str, train_fraction, seed, step)
model = Model(data)
ares = ARES(data, model)

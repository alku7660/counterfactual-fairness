import numpy as np
import pandas as pd
from itertools import product
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, tuplelist
from data_constructor import load_dataset
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

    def __init__(self, data) -> None:
        self.discretized_train_df = data.discretized_train_df
        self.protected_groups = data.feat_protected
        self.sensitive_groups = self.get_sensitive_groups()
        self.apriori_df = self.get_apriori_df()
    
    def get_apriori_df(self):
        """
        Obtains the apriori conjunction predicates from the frequent itemsets from the apriori algorithm, as explained in:
        Rawal, Kaivalya, and Himabindu Lakkaraju. "Beyond individualized recourse: Interpretable and interactive summaries of actionable recourses." Advances in Neural Information Processing Systems 33 (2020): 12187-12198.
        """
        apriori_df = apriori(self.discretized_train_df, min_support=0.05, use_colnames=True)
        return apriori_df

    def get_sensitive_groups(self):
        """
        Obtains a list of sensitive groups
        """
        sensitive_group_list = []
        for sensitive_group in self.protected_groups.keys:
            sensitive_group_list.extend([x for x in self.discretized_train_df.columns if sensitive_group in x])
        return sensitive_group_list 

    def get_recourse_predicates_per_sensitive_group(self):
        """
        Obtains a Dict object containing the frequent itemsets (recourse predicates) for each of the sensitive groups.
        """
        predicate_dict = {}

        return predicate_dict

data_str = 'synthetic_athlete'
train_fraction = 0.7
seed = 12345
step = 0.01
data = load_dataset(data_str, train_fraction, seed, step)
ares = ARES(data)

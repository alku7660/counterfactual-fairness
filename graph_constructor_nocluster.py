import numpy as np
import pandas as pd
from itertools import product
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, tuplelist
from scipy.spatial import distance_matrix
from evaluator_constructor import distance_calculation, verify_feasibility
from centroid_constructor import inverse_transform_original
from nnt import nn_for_juice
import time
from scipy.stats import norm
import copy

class Graph:

    def __init__(self, data, model, feat, feat_values, sensitive_group_dict, type, percentage) -> None:
        self.percentage = percentage
        self.feat = feat
        self.sensitive_group_dict = sensitive_group_dict
        self.sensitive_feature_instances, self.idx_feat_value_dict = self.find_sensitive_feat_instances(data, feat_values)
        self.ioi_label = data.undesired_class
        self.train_cf = self.find_train_cf(data, model, feat_values, type)
        self.epsilon = self.get_epsilon(data, dist=type)
        self.feat_possible_values, self.all_nodes, self.C, self.W, self.CW, self.F, self.rho, self.eta = self.construct_graph(data, model, feat_values, type)

    def find_sensitive_group_instances(self, data, feat_val):
        """
        Finds the instances of the sensitive group given as parameter by index
        """
        sensitive_group_idx = self.sensitive_group_dict[feat_val]
        sensitive_group_instances = data.transformed_false_undesired_test_df.loc[sensitive_group_idx].values
        return sensitive_group_instances
    
    def find_sensitive_feat_instances(self, data, feat_values):
        """
        Finds the instances of the sensitive feature by index
        """
        sensitive_feature_instances, idx_feat_value_dict = [], {}
        start_idx = 0
        for feat_value in feat_values:
            sensitive_group_instances = self.find_sensitive_group_instances(data, feat_value)
            sensitive_feature_instances.append(sensitive_group_instances)
            for idx in range(start_idx, len(sensitive_group_instances)): 
                idx_feat_value_dict[idx] = feat_value
            start_idx += len(sensitive_group_instances)
        sensitive_feature_instances = np.array(sensitive_feature_instances)
        return sensitive_feature_instances, idx_feat_value_dict
    
    def estimate_sensitive_group_positive(self, data, feat_val):
        """
        Estimates the amount of ground truth positives in the feature sensitive group given as parameter
        """
        sensitive_group_df = data.test_df.loc[(data.test_df[self.feat] == feat_val) & (data.test_target == data.desired_class)]
        return len(sensitive_group_df)
    
    def find_train_specific_feature_val(self, data, feat_value):
        """
        Finds all the training observations belonging to the feature value of interest
        """
        train_target_df = copy.deepcopy(data.train_df)
        train_target_df['target'] = data.train_target
        train_target_feat_val_df = train_target_df[train_target_df[self.feat] == feat_value]
        target_feat_val = train_target_feat_val_df['target'].values
        del train_target_feat_val_df['target']
        train_feat_val_np = data.transform_data(train_target_feat_val_df).values
        return train_target_feat_val_df, target_feat_val, train_feat_val_np

    def find_train_desired_label(self, train_np, train_target, train_pred, extra_search):
        """
        Finds the training instances that have the desired label from either ground truth and/or prediction
        """
        if not extra_search:
            train_cf = train_np[(train_target != self.ioi_label) & (train_pred != self.ioi_label)]
        else:
            train_cf = train_np[train_target != self.ioi_label]
        return train_cf

    def find_closest_train_cf_per_instance(self, data, normal_instance, train_cfs, type, extra_search):
        """
        Finds the closest counterfactual training instance to the instance of interest 
        """
        list_instances = []
        for train_i in range(train_cfs.shape[0]):
            train_cfs = train_cfs[train_i]
            if extra_search:
                if verify_feasibility(normal_instance, train_cfs[train_i], data):
                    dist = distance_calculation(train_cfs[train_i], normal_instance, data, type=type)
                    list_instances.append((train_cfs[train_i], dist))
            else:
                dist = distance_calculation(train_cfs[train_i], normal_instance, data, type=type)
                list_instances.append((train_cfs[train_i], dist))
        list_instances.sort(key=lambda x: x[1])
        list_instances = [i[0] for i in list_instances]
        closest_train_cf = list_instances[0]
        return closest_train_cf

    def find_train_cf(self, data, model, feat_values, type, extra_search=False):
        """
        Finds the set of training observations belonging to, and predicted as, the counterfactual class and that belong to the same sensitive group as the centroid (this avoids node generation explosion)
        """
        train_cf_list = []
        for feat_value in feat_values:
            train_feat_val_df, target_feat_val, train_feat_val_np = self.find_train_specific_feature_val(data, feat_value)
            train_np_feat_val_pred = model.model.predict(train_feat_val_np)
            train_desired_label_np = self.find_train_desired_label(train_feat_val_np, target_feat_val, train_np_feat_val_pred, extra_search)
            sensitive_group_instances = self.find_sensitive_group_instances(data, feat_value)
            for instance in sensitive_group_instances:
                closest_train_cf = self.find_closest_train_cf_per_instance(data, instance, train_desired_label_np, type, extra_search)
                if not any(np.array_equal(closest_train_cf, x) for x in train_cf_list):
                    train_cf_list.append(closest_train_cf)
            # train_cf_list = train_cf_list[:int(len(train_cf_list)*self.percentage)]
        return train_cf_list

    def construct_graph(self, data, feat_values, model, type):
        """
        Constructs the graph and the required parameters to run Fijuice several lagrange values
        """
        print(f'Obtained all training CF: {len(self.train_cf)}')
        feat_possible_values = self.get_feat_possible_values(data)
        print(f'Obtained all possible feature values from training CF')
        graph_nodes = self.get_graph_nodes(data, model, feat_possible_values)
        all_nodes = self.train_cf + graph_nodes
        print(f'Obtained all possible nodes in the graph: {len(all_nodes)}')
        C, W, CW = self.get_all_costs_weights(data, type, feat_values, all_nodes)
        print(f'Obtained all costs in the graph')
        F = self.get_all_feasibility(data, all_nodes)
        print(f'Obtained all feasibility in the graph')
        rho = self.get_all_likelihood(data, all_nodes, dist=type)
        print(f'Obtained all Likelihood parameter')
        eta = self.get_all_effectiveness(data, feat_values, all_nodes)
        print(f'Obtained all effectiveness parameter')
        return feat_possible_values, all_nodes, C, W, CW, F, rho, eta
    
    def get_feat_possible_values(self, data, obj=None, points=None):
        """
        Method that obtains the features possible values
        """
        if obj is None:
            sensitive_feature_instances = self.sensitive_feature_instances
        else:
            sensitive_feature_instances = obj
        if points is None:
            points = self.train_cf
        else:
            points = points
        feat_possible_values_all_centroids = {}
        for instance_idx in range(len(sensitive_feature_instances)):
            instance = sensitive_feature_instances[instance_idx]
            cf_feat_possible_values = {}
            for k in range(len(points)):
                train_cf_k = points[k]
                v = instance - train_cf_k
                nonzero_index = list(np.nonzero(v)[0])
                feat_checked = []
                feat_possible_values = []
                for i in range(len(instance)):
                    if i not in feat_checked:
                        feat_i = data.processed_features[i]
                        if feat_i in data.bin_enc_cols:
                            if i in nonzero_index:
                                value = [train_cf_k[i], instance[i]]
                            else:
                                value = [train_cf_k[i]]
                            feat_checked.extend([i])
                        elif feat_i in data.cat_enc_cols:
                            idx_cat_i = data.idx_cat_cols_dict[feat_i[:-4]]
                            nn_cat_idx = list(train_cf_k[idx_cat_i])
                            if any(item in idx_cat_i for item in nonzero_index):
                                ioi_cat_idx = list(instance[idx_cat_i])
                                value = [nn_cat_idx, ioi_cat_idx]
                            else:
                                value = [nn_cat_idx]
                            feat_checked.extend(idx_cat_i)
                        elif feat_i in data.ordinal:
                            if i in nonzero_index:
                                values_i = list(data.processed_feat_dist[feat_i].keys())
                                max_val_i, min_val_i = max(instance[i], train_cf_k[i]), min(instance[i], train_cf_k[i])
                                value = [j for j in values_i if j <= max_val_i and j >= min_val_i]
                            else:
                                value = [train_cf_k[i]]
                            feat_checked.extend([i])
                        elif feat_i in data.continuous:
                            if i in nonzero_index:
                                max_val_i, min_val_i = max(instance[i], train_cf_k[i]), min(instance[i], train_cf_k[i])
                                value = self.continuous_feat_values(i, min_val_i, max_val_i, data)
                            else:
                                value = [train_cf_k[i]]
                            feat_checked.extend([i])
                        feat_possible_values.append(value)
                cf_feat_possible_values[k] = feat_possible_values
            feat_possible_values_all_centroids[instance_idx] = cf_feat_possible_values
        return feat_possible_values_all_centroids

    def make_array(self, i):
        """
        Method that transforms a generator instance into array  
        """
        list_i = list(i)
        new_list = []
        for j in list_i:
            if isinstance(j, list):
                new_list.extend([k for k in j])
            else:
                new_list.extend([j])
        return np.array(new_list)

    def get_graph_nodes(self, data, model, feat_possible_values):
        """
        Generator that contains all the nodes located in the space between the training CFs and the normal_ioi (all possible, CF-labeled nodes)
        """
        graph_nodes = []
        for instance_idx in range(len(self.sensitive_feature_instances)):
            for k in range(len(self.train_cf)):
                print(f'Graph: sensitive group instance {instance_idx} and training CF {k}. Nodes size: {len(graph_nodes)}')
                feat_possible_values_k = feat_possible_values[instance_idx][k]
                permutations = product(*feat_possible_values_k)
                for i in permutations:
                    perm_i = self.make_array(i)
                    if model.model.predict(perm_i.reshape(1, -1)) != self.ioi_label and \
                        not any(np.array_equal(perm_i, x) for x in graph_nodes) and \
                        not any(np.array_equal(perm_i, x) for x in self.train_cf):
                        graph_nodes.append(perm_i)
        return graph_nodes

    def get_all_costs_weights(self, data, type, feat_values, all_nodes):
        """
        Method that outputs the cost parameters required for optimization
        """
        C = {}
        for instance_idx in range(1, len(self.sensitive_feature_instances) + 1):
            instance = self.sensitive_feature_instances[instance_idx - 1]
            feat_value = self.idx_feat_value_dict[instance_idx - 1]
            len_positives_sensitive_group = self.estimate_sensitive_group_positive(data, feat_value)
            for k in range(1, len(all_nodes) + 1):
                node_k = all_nodes[k - 1]
                dist_instance_node = distance_calculation(instance, node_k, data, type)
                C[instance_idx, k] = dist_instance_node/(2*len_positives_sensitive_group)
                print(f'Costs for centroid {instance_idx}, node {k} calculated')
        return C

    def get_all_feasibility(self, data, all_nodes):
        """
        Outputs the counterfactual feasibility parameter for all graph nodes (including the training CFs) 
        """
        F = {}
        for instance_idx in range(1, len(self.sensitive_feature_instances) + 1):
            instance = self.sensitive_feature_instances[instance_idx - 1]
            for k in range(1, len(all_nodes) + 1):
                node_k = all_nodes[k - 1]
                F[instance_idx, k] = verify_feasibility(instance, node_k, data)
        return F
    
    def get_all_effectiveness(self, data, feat_values, all_nodes):
        """
        Outputs the counterfactual effectiveness parameter for all nodes (including the training CFs)
        """
        eta = {}
        for feat_value in feat_values:
            feat_value_idx = self.sensitive_group_dict[feat_value]
            len_sensitive_group = len(feat_value_idx)
            for k in range(1, len(all_nodes) + 1):
                sum_eta = 0
                node_k = all_nodes[k - 1]
                for instance_idx in feat_value_idx:
                    instance = data.transformed_false_undesired_test_df[instance_idx].values
                    sum_eta += verify_feasibility(instance, node_k, data)
                eta[k] = sum_eta/len_sensitive_group
            print(f'Highest eta value for {data.feat_protected[self.feat][feat_value]} : {np.max(list(eta.values()))}')
        return eta

    def continuous_feat_values(self, i, min_val, max_val, data):
        """
        Method that defines how to discretize the continuous features
        """
        sorted_feat_i = list(np.sort(data.transformed_train_np[:,i][(data.transformed_train_np[:,i] >= min_val) & (data.transformed_train_np[:,i] <= max_val)]))
        value = list(np.unique(sorted_feat_i))
        if len(value) <= 10:
            if min_val not in value:
                value = [min_val] + value
            if max_val not in value:
                value = value + [max_val]
            return value
        else:
            mean_val, std_val = np.mean(data.transformed_train_np[:,i]), np.std(data.transformed_train_np[:,i])
            percentiles_range = list(np.linspace(0, 1, 11))
            value = []
            for perc in percentiles_range:
                value.append(norm.ppf(perc, loc=mean_val, scale=std_val))
            value = [val for val in value if val >= min_val and val <= max_val]
            if min_val not in value:
                value = [min_val] + value
            if max_val not in value:
                value = value + [max_val]
        return value
    
    def get_epsilon(self, data, dist='euclidean'):
        """
        Calculates the distance 
        """
        distance = distance_matrix(data.transformed_train_np, data.transformed_train_np, p=1)
        upper_tri_distance = distance[np.triu_indices(len(data.transformed_train_np), k = 1)]
        return np.std(upper_tri_distance, ddof=1) 

    def get_all_likelihood(self, data, all_nodes, dist='euclidean'):
        """
        Extracts the likelihood of all the nodes obtained
        """
        rho = {}
        distance = distance_matrix(all_nodes, data.transformed_train_np, p=1)
        gaussian_kernel = np.exp(-(distance/self.epsilon)**2)
        sum_gaussian_kernel_col = np.sum(gaussian_kernel, axis=1)
        max_sum_gaussian_kernel_col = np.max(sum_gaussian_kernel_col)
        min_sum_gaussian_kernel_col = np.min(sum_gaussian_kernel_col)
        for i in range(1, len(all_nodes) + 1):
            rho[i] = (sum_gaussian_kernel_col[i-1] - min_sum_gaussian_kernel_col)/(max_sum_gaussian_kernel_col - min_sum_gaussian_kernel_col)
        return rho
            

            
            

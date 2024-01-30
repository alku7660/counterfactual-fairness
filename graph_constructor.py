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

    def __init__(self, data, model, cluster, feat, type, percentage) -> None:
        self.percentage = percentage
        self.cluster = cluster
        self.feat = feat
        self.feature_centroids, self.feature_clusters, self.feature_groups = self.select_centroids_for_feature()
        self.ioi_label = cluster.undesired_class
        self.train_cf = self.find_train_cf(data, model, type)
        self.epsilon = self.get_epsilon(data, dist=type)
        self.feat_possible_values, self.all_nodes, self.C, self.W, self.CW, self.F, self.rho, self.eta = self.construct_graph(data, model, type)
    
    def select_centroids_clusters_groups_for_feature(self):
        """
        Selects only the centroids for the feature of interest
        """
        feature_centroids, feature_clusters, feature_groups = [], [], []
        for idx in range(len(self.cluster.filtered_centroids_list)):
            c = self.cluster.filtered_centroids_list[idx]
            cluster_list = self.cluster.filtered_clusters_list[idx]
            c_group = self.cluster.group_dict[idx + 1] 
            feature = c.feat
            if feature != self.feat:
                continue
            else:
                feature_centroids.append(c)
                feature_clusters.append(cluster_list)
                feature_groups.append(c_group)
        return feature_centroids, feature_clusters, feature_groups

    def find_train_specific_feature_val(self, data, feat_val):
        """
        Finds all the training observations belonging to the feature value of interest
        """
        train_target_df = copy.deepcopy(data.train_df)
        train_target_df['target'] = data.train_target
        train_target_feat_val_df = train_target_df[train_target_df[self.feat] == feat_val]
        target_feat_val = train_target_feat_val_df['target'].values()
        del train_target_feat_val_df['target']
        train_feat_val_np = data.transform_data(train_target_feat_val_df)
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

    def find_train_cf(self, data, model, type, extra_search=False):
        """
        Finds the set of training observations belonging to, and predicted as, the counterfactual class and that belong to the same sensitive group as the centroid (this avoids node generation explosion)
        """
        sort_train_cf_centroid = []
        for idx in range(self.feature_centroids):
            c = self.feature_centroids[idx]
            feat_val = c.feat_val
            train_feat_val_df, target_feat_val, train_feat_val_np = self.find_train_specific_feature_val(data, feat_val)
            train_np_feat_val_pred = model.model.predict(train_feat_val_np)
            train_desired_label_np = self.find_train_desired_label(train_feat_val_np, target_feat_val, train_np_feat_val_pred, extra_search)
            normal_centroid = c.normal_x
            for i in range(train_desired_label_np.shape[0]):
                if extra_search:
                    if verify_feasibility(normal_centroid, train_desired_label_np[i], data):
                        dist = distance_calculation(train_desired_label_np[i], normal_centroid, data, type=type)
                        sort_train_cf_centroid.append((train_desired_label_np[i], dist))
                else:
                    dist = distance_calculation(train_desired_label_np[i], normal_centroid, data, type=type)
                    sort_train_cf_centroid.append((train_desired_label_np[i], dist))
        sort_train_cf_centroid.sort(key=lambda x: x[1])
        sort_train_cf_centroid = [i[0] for i in sort_train_cf_centroid]
        sort_train_cf_centroid = sort_train_cf_centroid[:int(len(sort_train_cf_centroid)*self.percentage)]
        return sort_train_cf_centroid

    def nn_list(self, data):
        """
        Method that gets the list of training observations labeled as cf-label with respect to the cf, ordered based on graph nodes size
        """
        permutations_train_cf_all = []
        for c in range(len(self.cluster.filtered_centroids_list)):
            centroid = self.cluster.filtered_centroids_list[c]
            c_justifiers_list = self.train_cf.iloc[c]['justifiers']
            permutations_train_cf = []
            for i in range(len(c_justifiers_list)):
                possible_feat_values_justifier_i = self.get_feat_possible_values(data, obj=[centroid], points=[c_justifiers_list[i]])[0][0]
                len_permutations = len(list(product(*possible_feat_values_justifier_i)))
                permutations_train_cf.append((c_justifiers_list[i], len_permutations))
                # print(f'Justifier {i+1}: Length permutations: {len_permutations}')
            permutations_train_cf.sort(key=lambda x: x[1])
            permutations_train_cf = [i[0] for i in permutations_train_cf]
            if len(permutations_train_cf) > self.k:
                permutations_train_cf = permutations_train_cf[:self.k]
            permutations_train_cf_all.extend(permutations_train_cf)
        return permutations_train_cf_all

    def construct_graph(self, data, model, type):
        """
        Constructs the graph and the required parameters to run Fijuice several lagrange values
        """
        print(f'Obtained all training CF: {len(self.train_cf)}')
        feat_possible_values = self.get_feat_possible_values(data)
        print(f'Obtained all possible feature values from training CF')
        graph_nodes = self.get_graph_nodes(data, model, feat_possible_values)
        all_nodes = self.train_cf + graph_nodes
        print(f'Obtained all possible nodes in the graph: {len(all_nodes)}')
        C, W, CW = self.get_all_costs_weights(data, type, all_nodes)
        print(f'Obtained all costs in the graph')
        F = self.get_all_feasibility(data, all_nodes)
        print(f'Obtained all feasibility in the graph')
        rho = self.get_all_likelihood(data, all_nodes, dist=type)
        print(f'Obtained all Likelihood parameter')
        eta = self.get_all_effectiveness(data, all_nodes)
        print(f'Obtained all effectiveness parameter')
        return feat_possible_values, all_nodes, C, W, CW, F, rho, eta
    
    def get_feat_possible_values(self, data, obj=None, points=None):
        """
        Method that obtains the features possible values
        """
        if obj is None:
            normal_centroids = self.feature_centroids
        else:
            normal_centroids = obj
        if points is None:
            points = self.train_cf
        else:
            points = points
        feat_possible_values_all_centroids = {}
        for c_idx in range(len(normal_centroids)):
            cf_feat_possible_values = {}
            normal_centroid = normal_centroids[c_idx].normal_x
            for k in range(len(points)):
                train_cf_k = points[k]
                v = normal_centroid - train_cf_k
                nonzero_index = list(np.nonzero(v)[0])
                feat_checked = []
                feat_possible_values = []
                for i in range(len(normal_centroid)):
                    if i not in feat_checked:
                        feat_i = data.processed_features[i]
                        if feat_i in data.bin_enc_cols:
                            if i in nonzero_index:
                                value = [train_cf_k[i], normal_centroid[i]]
                            else:
                                value = [train_cf_k[i]]
                            feat_checked.extend([i])
                        elif feat_i in data.cat_enc_cols:
                            idx_cat_i = data.idx_cat_cols_dict[feat_i[:-4]]
                            nn_cat_idx = list(train_cf_k[idx_cat_i])
                            if any(item in idx_cat_i for item in nonzero_index):
                                ioi_cat_idx = list(normal_centroid[idx_cat_i])
                                value = [nn_cat_idx, ioi_cat_idx]
                            else:
                                value = [nn_cat_idx]
                            feat_checked.extend(idx_cat_i)
                        elif feat_i in data.ordinal:
                            if i in nonzero_index:
                                values_i = list(data.processed_feat_dist[feat_i].keys())
                                max_val_i, min_val_i = max(normal_centroid[i], train_cf_k[i]), min(normal_centroid[i], train_cf_k[i])
                                value = [j for j in values_i if j <= max_val_i and j >= min_val_i]
                            else:
                                value = [train_cf_k[i]]
                            feat_checked.extend([i])
                        elif feat_i in data.continuous:
                            if i in nonzero_index:
                                max_val_i, min_val_i = max(normal_centroid[i], train_cf_k[i]), min(normal_centroid[i], train_cf_k[i])
                                value = self.continuous_feat_values(i, min_val_i, max_val_i, data)
                            else:
                                value = [train_cf_k[i]]
                            feat_checked.extend([i])
                        feat_possible_values.append(value)
                cf_feat_possible_values[k] = feat_possible_values
            feat_possible_values_all_centroids[c_idx] = cf_feat_possible_values
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
        for c_idx in range(len(self.feature_centroids)):
            # print(f'Analyzing centroid {c_idx} for graph nodes...')
            for k in range(len(self.train_cf)):
                print(f'Analyzing centroid {c_idx} and training CF {k} for graph nodes. Current length of nodes: {len(graph_nodes)}')
                feat_possible_values_k = feat_possible_values[c_idx][k]
                normal_centroid = self.feature_centroids[c_idx].normal_x
                permutations = product(*feat_possible_values_k)
                for i in permutations:
                    perm_i = self.make_array(i)
                    if model.model.predict(perm_i.reshape(1, -1)) != self.ioi_label and \
                        not any(np.array_equal(perm_i, x) for x in graph_nodes) and \
                        not any(np.array_equal(perm_i, x) for x in self.train_cf) and \
                        verify_feasibility(normal_centroid, perm_i, data):
                        graph_nodes.append(perm_i)
        return graph_nodes
    
    def get_all_costs_weights(self, data, type, all_nodes):
        """
        Method that outputs the cost parameters required for optimization
        """
        C, W, CW = {}, {}, {}
        clusters_total_instances = 0
        for c_idx in range(1, len(self.feature_centroids) + 1):
            clusters_total_instances += self.feature_centroids[c_idx - 1].cluster_size
        for c_idx in range(1, len(self.feature_centroids) + 1):
            cluster_instances_list = self.feature_clusters[c_idx - 1]
            W[c_idx] = len(cluster_instances_list)/clusters_total_instances
            for k in range(1, len(all_nodes) + 1):
                node_k = all_nodes[k - 1]
                dist_instance_node = 0
                for instance_idx in cluster_instances_list:
                    instance = self.cluster.transformed_false_undesired_test_df.loc[instance_idx].values
                    dist_instance_node += distance_calculation(instance, node_k, data, type)
                C[c_idx, k] = dist_instance_node/len(cluster_instances_list)
                CW[c_idx, k] = C[c_idx, k]*W[c_idx]
                print(f'Costs for centroid {c_idx}, node {k} calculated')
        return C, W, CW
    
    def get_all_feasibility(self, data, all_nodes):
        """
        Outputs the counterfactual feasibility parameter for all graph nodes (including the training CFs) 
        """
        F = {}
        for c_idx in range(1, len(self.feature_centroids) + 1):
            normal_centroid = self.feature_centroids[c_idx - 1].normal_x
            for k in range(1, len(all_nodes) + 1):
                node_k = all_nodes[k - 1]
                F[c_idx, k] = verify_feasibility(normal_centroid, node_k, data)
        return F
    
    def get_all_effectiveness(self, data, all_nodes):
        """
        Outputs the counterfactual effectiveness parameter for all nodes (including the training CFs)
        """
        eta = {}
        len_cluster_instances = 0
        for c_idx in range(1, len(self.feature_centroids) + 1):
            len_cluster_instances += self.feature_centroids[c_idx - 1].cluster_size
        for k in range(1, len(all_nodes) + 1):
            sum_eta = 0
            node_k = all_nodes[k - 1]
            for c_idx in range(1, len(self.feature_centroids) + 1):
                cluster_instances_list = self.feature_clusters[c_idx - 1]
                for instance_idx in cluster_instances_list:
                    instance = self.cluster.transformed_false_undesired_test_df.loc[instance_idx].values
                    sum_eta += verify_feasibility(instance, node_k, data)
            eta[k] = sum_eta/len_cluster_instances
            print(f'Highest eta value: {np.max(list(eta.values()))}')
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
            

            
            

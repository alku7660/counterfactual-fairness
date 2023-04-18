import numpy as np
import pandas as pd
from itertools import product
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, tuplelist
from evaluator_constructor import distance_calculation, verify_feasibility
from nnt import nn_for_juice
import time
from scipy.stats import norm

class Graph:

    def __init__(self, data, model, cluster, type, t=100, k=10) -> None:
        self.cluster = cluster
        self.ioi_label = cluster.undesired_class
        self.t = t
        self.k = k
        self.potential_justifiers = self.find_potential_justifiers(data, model, type)
        self.potential_justifiers = self.nn_list(data)
        self.pot_justifier_feat_possible_values, self.all_nodes, self.C, self.W, self.CW, self.F, self.A = self.construct_graph(data, model, type)
    
    def find_potential_justifiers(self, data, model, type, ijuice_search=False):
        """
        Finds the set of training observations belonging to, and predicted as, the counterfactual class
        """
        train_np = data.transformed_train_np
        train_target = data.train_target
        train_pred = model.model.predict(train_np)
        if not ijuice_search:
            potential_justifiers = train_np[(train_target != self.ioi_label) & (train_pred != self.ioi_label)]
        else:
            potential_justifiers = train_np[train_target != self.ioi_label]
        potential_justifiers_df = pd.DataFrame(columns = ['centroid','feat','feat_val','justifiers'])
        for idx in range(len(self.cluster.filtered_centroids_list)):
            c = self.cluster.filtered_centroids_list[idx]
            feat = c.feat
            feat_val = c.feat_val
            normal_centroid = c.normal_x
            sort_potential_justifiers_centroid = []
            for i in range(potential_justifiers.shape[0]):
                if ijuice_search: 
                    if verify_feasibility(normal_centroid, potential_justifiers[i], data):
                        dist = distance_calculation(potential_justifiers[i], normal_centroid, data, type=type)
                        sort_potential_justifiers_centroid.append((potential_justifiers[i], dist))
                else:
                    dist = distance_calculation(potential_justifiers[i], normal_centroid, data, type=type)
                    sort_potential_justifiers_centroid.append((potential_justifiers[i], dist))
            sort_potential_justifiers_centroid.sort(key=lambda x: x[1])
            sort_potential_justifiers_centroid = [i[0] for i in sort_potential_justifiers_centroid]
            if len(sort_potential_justifiers_centroid) > self.t:
                sort_potential_justifiers_centroid = sort_potential_justifiers_centroid[:self.t]
            centroid_df_data = pd.DataFrame([[normal_centroid, feat, feat_val, sort_potential_justifiers_centroid]], index=[idx], columns=potential_justifiers_df.columns)
            potential_justifiers_df = pd.concat((potential_justifiers_df, centroid_df_data), axis=0)
        return potential_justifiers_df

    def nn_list(self, data):
        """
        Method that gets the list of training observations labeled as cf-label with respect to the cf, ordered based on graph nodes size
        """
        permutations_potential_justifiers_all = []
        for c in range(len(self.cluster.filtered_centroids_list)):
            centroid = self.cluster.filtered_centroids_list[c]
            c_justifiers_list = self.potential_justifiers.iloc[c]['justifiers']
            permutations_potential_justifiers = []
            for i in range(len(c_justifiers_list)):
                possible_feat_values_justifier_i = self.get_feat_possible_values(data, obj=[centroid], points=[c_justifiers_list[i]])[0][0]
                len_permutations = len(list(product(*possible_feat_values_justifier_i)))
                permutations_potential_justifiers.append((c_justifiers_list[i], len_permutations))
                # print(f'Justifier {i+1}: Length permutations: {len_permutations}')
            permutations_potential_justifiers.sort(key=lambda x: x[1])
            permutations_potential_justifiers = [i[0] for i in permutations_potential_justifiers]
            if len(permutations_potential_justifiers) > self.k:
                permutations_potential_justifiers = permutations_potential_justifiers[:self.k]
            permutations_potential_justifiers_all.extend(permutations_potential_justifiers)
        return permutations_potential_justifiers_all

    def construct_graph(self, data, model, type):
        """
        Constructs the graph and the required parameters to run Fijuice several lagrange values
        """
        print(f'Obtained all potential justifiers: {len(self.potential_justifiers)}')
        pot_justifier_feat_possible_values = self.get_feat_possible_values(data)
        print(f'Obtained all possible feature values from potential justifiers')
        graph_nodes = self.get_graph_nodes(model, pot_justifier_feat_possible_values)
        all_nodes = self.potential_justifiers + graph_nodes
        print(f'Obtained all possible nodes in the graph: {len(all_nodes)}')
        C, W, CW = self.get_all_costs_weights(data, type, all_nodes)
        print(f'Obtained all costs in the graph')
        F = self.get_all_feasibility(data, all_nodes)
        print(f'Obtained all feasibility in the graph')
        A = self.get_all_adjacency(data, all_nodes)
        print(f'Obtained adjacency matrix')
        return pot_justifier_feat_possible_values, all_nodes, C, W, CW, F, A
    
    def get_feat_possible_values(self, data, obj=None, points=None):
        """
        Method that obtains the features possible values
        """
        if obj is None:
            normal_centroids = self.cluster.filtered_centroids_list
        else:
            normal_centroids = obj
        if points is None:
            points = self.potential_justifiers
        else:
            points = points
        pot_justifier_feat_possible_values_all_centroids = {}
        for c_idx in range(len(normal_centroids)):
            pot_justifier_feat_possible_values = {}
            normal_centroid = normal_centroids[c_idx].normal_x
            for k in range(len(points)):
                potential_justifier_k = points[k]
                v = normal_centroid - potential_justifier_k
                nonzero_index = list(np.nonzero(v)[0])
                feat_checked = []
                feat_possible_values = []
                for i in range(len(normal_centroid)):
                    if i not in feat_checked:
                        feat_i = data.processed_features[i]
                        if feat_i in data.bin_enc_cols:
                            if i in nonzero_index:
                                value = [potential_justifier_k[i], normal_centroid[i]]
                            else:
                                value = [potential_justifier_k[i]]
                            feat_checked.extend([i])
                        elif feat_i in data.cat_enc_cols:
                            idx_cat_i = data.idx_cat_cols_dict[feat_i[:-4]]
                            nn_cat_idx = list(potential_justifier_k[idx_cat_i])
                            if any(item in idx_cat_i for item in nonzero_index):
                                ioi_cat_idx = list(normal_centroid[idx_cat_i])
                                value = [nn_cat_idx, ioi_cat_idx]
                            else:
                                value = [nn_cat_idx]
                            feat_checked.extend(idx_cat_i)
                        elif feat_i in data.ordinal:
                            if i in nonzero_index:
                                values_i = list(data.processed_feat_dist[feat_i].keys())
                                max_val_i, min_val_i = max(normal_centroid[i], potential_justifier_k[i]), min(normal_centroid[i], potential_justifier_k[i])
                                value = [j for j in values_i if j <= max_val_i and j >= min_val_i]
                            else:
                                value = [potential_justifier_k[i]]
                            feat_checked.extend([i])
                        elif feat_i in data.continuous:
                            if i in nonzero_index:
                                max_val_i, min_val_i = max(normal_centroid[i], potential_justifier_k[i]), min(normal_centroid[i], potential_justifier_k[i])
                                value = self.continuous_feat_values(i, min_val_i, max_val_i, data)
                            else:
                                value = [potential_justifier_k[i]]
                            feat_checked.extend([i])
                        feat_possible_values.append(value)
                pot_justifier_feat_possible_values[k] = feat_possible_values
            pot_justifier_feat_possible_values_all_centroids[c_idx] = pot_justifier_feat_possible_values
        return pot_justifier_feat_possible_values_all_centroids
    
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

    def get_graph_nodes(self, model, pot_justifier_feat_possible_values):
        """
        Generator that contains all the nodes located in the space between the potential justifiers and the normal_ioi (all possible, CF-labeled nodes)
        """
        graph_nodes = []
        for c_idx in range(len(self.cluster.filtered_centroids_list)):
            # print(f'Analyzing centroid {c_idx} for graph nodes...')
            for k in range(len(self.potential_justifiers)):
                print(f'Analyzing centroid {c_idx} and potential justifier {k} for graph nodes...')
                feat_possible_values_k = pot_justifier_feat_possible_values[c_idx][k]
                permutations = product(*feat_possible_values_k)
                for i in permutations:
                    perm_i = self.make_array(i)
                    if model.model.predict(perm_i.reshape(1, -1)) != self.ioi_label and \
                        not any(np.array_equal(perm_i, x) for x in graph_nodes) and \
                        not any(np.array_equal(perm_i, x) for x in self.potential_justifiers):
                        graph_nodes.append(perm_i)
        return graph_nodes
    
    def get_all_costs_weights(self, data, type, all_nodes):
        """
        Method that outputs the cost parameters required for optimization
        """
        C, W, CW = {}, {}, {}
        clusters_total_instances = 0
        for c_idx in range(1, len(self.cluster.filtered_centroids_list) + 1):
            cluster_instances_list = self.cluster.filtered_clusters_list[c_idx - 1]
            clusters_total_instances += len(cluster_instances_list)
        for c_idx in range(1, len(self.cluster.filtered_centroids_list) + 1):
            normal_centroid = self.cluster.filtered_centroids_list[c_idx - 1].normal_x
            cluster_instances_list = self.cluster.filtered_clusters_list[c_idx - 1]
            W[c_idx] = len(cluster_instances_list)/clusters_total_instances
            for k in range(1, len(all_nodes) + 1):
                node_k = all_nodes[k-1]
                dist_instance_node = 0
                for instance_idx in cluster_instances_list:
                    instance = self.cluster.transformed_false_undesired_test_df.loc[instance_idx].values
                    dist_instance_node += distance_calculation(instance, node_k, data, type)
                C[c_idx, k] = dist_instance_node/len(cluster_instances_list)
                CW[c_idx, k] = C[c_idx, k]*W[c_idx]
        return C, W, CW
    
    def get_all_feasibility(self, data, all_nodes):
        """
        Outputs the counterfactual feasibility parameter for all graph nodes (including the potential justifiers) 
        """
        F = {}
        for c_idx in range(1, len(self.cluster.filtered_centroids_list) + 1):
            normal_centroid = self.cluster.filtered_centroids_list[c_idx - 1].normal_x
            for k in range(1, len(all_nodes) + 1):
                node_k = all_nodes[k-1]
                F[c_idx, k] = verify_feasibility(normal_centroid, node_k, data)
        return F
    
    def get_all_adjacency(self, data, all_nodes):
        """
        Method that outputs the adjacency matrix required for optimization
        """
        toler = 0.00001
        centroids_array = np.array([self.cluster.filtered_centroids_list[i].normal_x for i in range(len(self.cluster.filtered_centroids_list))])
        justifiers_array = np.array(self.potential_justifiers)
        A = tuplelist()
        for i in range(1, len(all_nodes) + 1):
            node_i = all_nodes[i - 1]
            for j in range(i + 1, len(all_nodes) + 1):
                node_j = all_nodes[j - 1]
                vector_ij = node_j - node_i
                nonzero_index = list(np.nonzero(vector_ij)[0])
                feat_nonzero = [data.processed_features[l] for l in nonzero_index]
                if len(nonzero_index) > 2:
                    continue
                elif len(nonzero_index) == 2:
                    if any(item in data.cat_enc_cols for item in feat_nonzero):
                        A.append((i,j))
                elif len(nonzero_index) == 1:
                    if any(item in data.ordinal for item in feat_nonzero):
                        if np.isclose(np.abs(vector_ij[nonzero_index]), data.feat_step[feat_nonzero], atol=toler).any():
                            A.append((i,j))
                    elif any(item in data.continuous for item in feat_nonzero):
                        max_val, min_val = float(max(max(centroids_array[:,nonzero_index]), max(justifiers_array[:,nonzero_index]))), float(min(min(centroids_array[:,nonzero_index]), min(justifiers_array[:,nonzero_index])))
                        values = self.continuous_feat_values(nonzero_index, min_val, max_val, data)
                        try:
                            value_node_i_idx = int(np.where(np.isclose(values, node_i[nonzero_index]))[0])
                            if value_node_i_idx > 0:
                                value_node_i_idx_inf = value_node_i_idx - 1
                                value_node_i_idx_sup = value_node_i_idx
                            else:
                                value_node_i_idx_inf = value_node_i_idx
                                value_node_i_idx_sup = value_node_i_idx + 1
                            if value_node_i_idx < len(values) - 1:
                                value_node_i_idx_inf = value_node_i_idx
                                value_node_i_idx_sup = value_node_i_idx + 1
                            else:
                                value_node_i_idx_inf = value_node_i_idx -1
                                value_node_i_idx_sup = value_node_i_idx
                        except:
                            if node_i[nonzero_index] < values[0]:
                                value_node_i_idx_inf, value_node_i_idx_sup = 0, 0
                            elif node_i[nonzero_index] > values[-1]:
                                value_node_i_idx_inf, value_node_i_idx_sup = len(values) - 1, len(values) - 1
                            for k in range(len(values) - 1):
                                if node_i[nonzero_index] <= values[k+1] and node_i[nonzero_index] >= values[k]:
                                    value_node_i_idx_inf, value_node_i_idx_sup = k, k+1  
                        close_node_j_values = [values[value_node_i_idx_inf], values[value_node_i_idx_sup]]
                        if any(np.isclose(node_j[nonzero_index], close_node_j_values)):
                            A.append((i,j))
                    elif any(item in data.binary for item in feat_nonzero):
                        if np.isclose(np.abs(vector_ij[nonzero_index]), [0,1], atol=toler).any():
                            A.append((i,j))
        return A
    
    def continuous_feat_values(self, i, min_val, max_val, data):
        """
        Method that defines how to discretize the continuous features
        """
        sorted_feat_i = list(np.sort(data.transformed_train_np[:,i][(data.transformed_train_np[:,i] >= min_val) & (data.transformed_train_np[:,i] <= max_val)]))
        value = list(np.unique(sorted_feat_i))
        if len(value) <= 100:
            if min_val not in value:
                value = [min_val] + value
            if max_val not in value:
                value = value + [max_val]
            return value
        else:
            mean_val, std_val = np.mean(data.transformed_train_np[:,i]), np.std(data.transformed_train_np[:,i])
            percentiles_range = list(np.linspace(0, 1, 101))
            value = []
            for perc in percentiles_range:
                value.append(norm.ppf(perc, loc=mean_val, scale=std_val))
            value = [val for val in value if val >= min_val and val <= max_val]
            if min_val not in value:
                value = [min_val] + value
            if max_val not in value:
                value = value + [max_val]
        return value
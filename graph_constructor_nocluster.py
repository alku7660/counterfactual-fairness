import numpy as np
import pandas as pd
from itertools import product
import networkx as nx
import gurobipy as gp
from gurobipy import GRB, tuplelist
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances
from evaluator_constructor import distance_calculation, verify_feasibility
from centroid_constructor import inverse_transform_original
from joblib import Parallel, delayed
from nnt import nn_for_juice
from itertools import chain

from sklearn.neighbors import NearestNeighbors
import time
from scipy.stats import norm
import copy

number_cores = -1

def find_sensitive_group_instances(data, feat_val, sensitive_group_dict):
    """
    Finds the instances of the sensitive group given as parameter by index
    """
    sensitive_group_idx = sensitive_group_dict[feat_val]
    sensitive_group_instances = data.transformed_false_undesired_test_df.loc[sensitive_group_idx]
    return sensitive_group_instances

def find_train_specific_feature_val(data, feat, feat_value):
    """
    Finds all the training observations belonging to the feature value of interest
    """
    train_target_df = copy.deepcopy(data.train_df)
    train_target_df['target'] = data.train_target
    train_target_feat_val_df = train_target_df[train_target_df[feat] == feat_value]
    target_feat_val = train_target_feat_val_df['target'].values
    del train_target_feat_val_df['target']
    train_feat_val_np = data.transform_data(train_target_feat_val_df).values
    return train_feat_val_np, target_feat_val

def find_train_desired_label(train_np, train_target, train_pred, extra_search, ioi_label):
    """
    Finds the training instances that have the desired label from either ground truth and/or prediction
    """
    if not extra_search:
        train_cf = train_np[(train_target != ioi_label) & (train_pred != ioi_label)]
    else:
        train_cf = train_np[train_target != ioi_label]
    return train_cf

def make_array(i):
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

def estimate_sensitive_group_positive(data, feat, feat_value):
    """
    Extracts length of the sensitive group test
    """
    sensitive_group_df = data.test_df.loc[(data.test_df[feat] == feat_value) & (data.test_target == data.desired_class)]
    return len(sensitive_group_df)

def get_all_costs_weights_parallel(data, feat, sensitive_feature_instances, all_nodes, instance_idx_to_original_idx_dict, sensitive_group_idx_feat_value_dict, instance_idx, k, type):
    """
    Parallelization of the cost calculation
    """
    instance = sensitive_feature_instances[instance_idx - 1]
    node_k = all_nodes[k - 1]
    distance = distance_calculation(instance, node_k, kwargs={'dat':data, 'type':type})
    original_instance_idx = instance_idx_to_original_idx_dict[instance_idx]
    feat_value = sensitive_group_idx_feat_value_dict[original_instance_idx]
    len_positives_sensitive_group = estimate_sensitive_group_positive(data, feat, feat_value)
    distance = distance/(len_positives_sensitive_group)
    return instance_idx, k, distance
    
def get_graph_nodes_parallel(data, model, sensitive_feature_instances, train_cf, min_closest_distance, feat_possible_values, k, instance_idx, ioi_label, type):
    """
    Parallelization of the graph nodes search
    """
    permutations_list = []
    feat_possible_values_k = feat_possible_values[instance_idx][k]
    permutations = product(*feat_possible_values_k)
    instance = sensitive_feature_instances[instance_idx]
    for i in permutations:
        perm_i = make_array(i)
        if verify_feasibility(instance, perm_i, data):
            if model.model.predict(perm_i.reshape(1, -1)) != ioi_label:
                if not any(np.array_equal(perm_i, x) for x in train_cf):
                    # if distance_calculation(instance, perm_i, kwargs={'dat':data, 'type':type}) < min_closest_distance:
                    permutations_list.append(perm_i)
    return permutations_list

def get_nearest_neighbor_parallel(data, model, feat, feat_value, extra_search, sensitive_group_dict, type):
    """
    Nearest neighbor parallelization
    """
    train_feat_val_np, target_feat_val = find_train_specific_feature_val(data, feat, feat_value)
    train_np_feat_val_pred = model.model.predict(train_feat_val_np)
    train_desired_label_np = find_train_desired_label(train_feat_val_np, target_feat_val, train_np_feat_val_pred, extra_search, data.undesired_class)
    sensitive_group_instances = find_sensitive_group_instances(data, feat_value, sensitive_group_dict).values
    neigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric=distance_calculation, metric_params={'dat':data, 'type':type}, n_jobs=1)
    neigh.fit(train_desired_label_np)
    print(f'NearestNeighbors fit for feat_value {feat_value}.')
    closest_distances, closest_cf_idx = neigh.kneighbors(sensitive_group_instances, return_distance=True)
    unique_closest_cf_idx = np.unique(closest_cf_idx).tolist()
    avg_closest_distance = np.mean(closest_distances)
    print(f'Found {len(unique_closest_cf_idx)} unique close training CF for feat_value {feat_value} with len instances {len(sensitive_group_instances)}')
    train_cf = train_desired_label_np[unique_closest_cf_idx]
    return train_cf, avg_closest_distance

class Graph:

    def __init__(self, data, model, feat, feat_values, sensitive_group_dict, type, percentage) -> None:
        self.percentage = percentage
        self.feat = feat
        self.sensitive_group_dict = sensitive_group_dict
        self.sensitive_feature_instances, self.sensitive_group_idx_feat_value_dict, self.instance_idx_to_original_idx_dict = self.find_sensitive_feat_instances(data, feat_values)
        self.ioi_label = data.undesired_class
        # self.train_cf = self.find_train_cf(data, model, feat_values, type)
        print('-------------------------------------------------------------------------')
        print('-------------Starting Nearest Training Counterfactual Search-------------')
        print('-------------------------------------------------------------------------')
        self.train_cf, self.min_closest_distance = self.nearest_neighbor_train_cf(data, model, feat, feat_values, type, sensitive_group_dict)
        print('-------------------------------------------------------------------------')
        print('----------------Finding Epsilon for Likelihood calculation---------------')
        print('-------------------------------------------------------------------------')
        self.epsilon = self.get_epsilon(data, dist=type)
        print('-------------------------------------------------------------------------')
        print('----------------------------Constructing Graph---------------------------')
        print('-------------------------------------------------------------------------')
        self.feat_possible_values, self.all_nodes, self.C, self.F, self.rho, self.eta = self.construct_graph(data, model, feat_values, type)

    def find_sensitive_group_instances(self, data, feat_val):
        """
        Finds the instances of the sensitive group given as parameter by index
        """
        sensitive_group_idx = self.sensitive_group_dict[feat_val]
        sensitive_group_instances = data.transformed_false_undesired_test_df.loc[sensitive_group_idx]
        return sensitive_group_instances
    
    def find_sensitive_feat_instances(self, data, feat_values):
        """
        Finds the instances of the sensitive feature by index
        """
        sensitive_feature_instances, sensitive_group_idx_feat_value_dict, instance_idx_to_original_idx_dict, counter = [], {}, {}, 1
        for feat_value in feat_values:
            sensitive_group_instances = self.find_sensitive_group_instances(data, feat_value)
            sensitive_group_instances_idx = sensitive_group_instances.index.to_list()
            sensitive_group_instances = sensitive_group_instances.values
            sensitive_feature_instances.append(sensitive_group_instances)
            for idx in sensitive_group_instances_idx: 
                sensitive_group_idx_feat_value_dict[idx] = feat_value
                instance_idx_to_original_idx_dict[counter] = idx
                counter += 1
        sensitive_feature_instances = np.concatenate(sensitive_feature_instances, axis=0)
        return sensitive_feature_instances, sensitive_group_idx_feat_value_dict, instance_idx_to_original_idx_dict
    
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
        return train_feat_val_np, target_feat_val

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
            train_cf_i = train_cfs[train_i]
            if extra_search:
                if verify_feasibility(normal_instance, train_cf_i, data):
                    dist = distance_calculation(train_cf_i, normal_instance, {'dat':data, 'type':type})
                    list_instances.append((train_cf_i, dist))
            else:
                dist = distance_calculation(train_cf_i, normal_instance, {'dat':data, 'type':type})
                list_instances.append((train_cf_i, dist))
        list_instances.sort(key=lambda x: x[1])
        list_instances = [i[0] for i in list_instances]
        closest_train_cf = list_instances[0]
        return closest_train_cf

    def find_train_cf(self, data, model, feat_values, type, extra_search=False):
        """
        Finds the set of training observations belonging to, and predicted as, the counterfactual class and that belong to the same sensitive group as the centroid (this avoids node generation explosion)
        """
        train_cf_list = []
        count_instances = 0
        for feat_value in feat_values:
            train_feat_val_np, target_feat_val = self.find_train_specific_feature_val(data, feat_value)
            train_np_feat_val_pred = model.model.predict(train_feat_val_np)
            train_desired_label_np = self.find_train_desired_label(train_feat_val_np, target_feat_val, train_np_feat_val_pred, extra_search)
            sensitive_group_instances = self.find_sensitive_group_instances(data, feat_value).values
            for instance in sensitive_group_instances:
                count_instances += 1
                start_time = time.time()
                closest_train_cf = self.find_closest_train_cf_per_instance(data, instance, train_desired_label_np, type, extra_search)
                end_time = time.time()
                if not any(np.array_equal(closest_train_cf, x) for x in train_cf_list):
                    train_cf_list.append(closest_train_cf)
                print(f'Count of instances {count_instances} out of {len(sensitive_group_instances)} in Feat {feat_value} (estimated total time: {len(sensitive_group_instances)*(end_time - start_time)*2})')
            train_cf_list = train_cf_list[:int(len(train_cf_list)*self.percentage)]
        return train_cf_list

    # def nearest_neighbor_train_cf(self, data, model, feat_values, type, extra_search=False):
    #     """
    #     Efficiently finds the set of training observations belonging to, and predicted as, the counterfactual class and that belong to the same sensitive group as the centroid (this avoids node generation explosion)
    #     """
    #     train_cf_list, closest_distances_list = [], []
    #     start_time = time.time()
    #     for feat_value in feat_values:
    #         train_feat_val_np, target_feat_val = self.find_train_specific_feature_val(data, feat_value)
    #         train_np_feat_val_pred = model.model.predict(train_feat_val_np)
    #         train_desired_label_np = self.find_train_desired_label(train_feat_val_np, target_feat_val, train_np_feat_val_pred, extra_search)
    #         sensitive_group_instances = self.find_sensitive_group_instances(data, feat_value).values
    #         neigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric=distance_calculation, metric_params={'dat':data, 'type':type}, n_jobs=1)
    #         neigh.fit(train_desired_label_np)
    #         print(f'NearestNeighbors fit for feat_value {feat_value}.')
    #         closest_distances, closest_cf_idx = neigh.kneighbors(sensitive_group_instances, return_distance=True)
    #         unique_closest_cf_idx = np.unique(closest_cf_idx).tolist()
    #         min_closest_distance = np.min(closest_distances)
    #         closest_distances_list.append(min_closest_distance) 
    #         print(f'Found {len(unique_closest_cf_idx)} unique close training CF for feat_value {feat_value} with len instances {len(sensitive_group_instances)}')
    #         # unique_closest_cf_idx_filtered = unique_closest_cf_idx[:int(len(unique_closest_cf_idx)*self.percentage)]
    #         # train_cf = train_desired_label_np[unique_closest_cf_idx_filtered]
    #         train_cf = train_desired_label_np[unique_closest_cf_idx]
    #         train_cf_list.append(train_cf)
    #     train_cf_array = np.concatenate(train_cf_list, axis=0)
    #     if data.name in ['synthetic_athlete','compass','german']:
    #         param_closest_distance = np.max(closest_distances)
    #     elif data.name in ['kdd_census']:
    #         param_closest_distance = np.mean(closest_distances)
    #     elif data.name in ['oulad','credit','bank','dutch','adult','student']:
    #         param_closest_distance = np.min(closest_distances)
    #     end_time = time.time()
    #     print(f'Found closest training CFs {len(train_cf_array)} for len instances {len(self.sensitive_feature_instances)}. (Total time: {(end_time - start_time)})')
    #     return train_cf_array, param_closest_distance

    def nearest_neighbor_train_cf(self, data, model, feat, feat_values, type, sensitive_group_dict, extra_search=False):
        """
        Efficiently finds the set of training observations belonging to, and predicted as, the counterfactual class and that belong to the same sensitive group as the centroid (this avoids node generation explosion)
        """
        start_time = time.time()
        results_list = Parallel(n_jobs=number_cores, verbose=10, prefer='processes')(delayed(get_nearest_neighbor_parallel)(data,
                                                                              model,
                                                                              feat,
                                                                              feat_value,
                                                                              extra_search,
                                                                              sensitive_group_dict,
                                                                              type
                                                                              ) for feat_value in feat_values) 
        train_cf_list, closest_distances = zip(*results_list)
        train_cf_array = np.concatenate(train_cf_list, axis=0)
        if data.name in ['synthetic_athlete','compass','german']:
            param_closest_distance = np.max(closest_distances)
        elif data.name in ['kdd_census']:
            param_closest_distance = np.mean(closest_distances)
        elif data.name in ['oulad','credit','bank','dutch','adult','student']:
            param_closest_distance = np.min(closest_distances)
        end_time = time.time()
        print(f'Found closest training CFs {len(train_cf_array)} for len instances {len(self.sensitive_feature_instances)}. (Total time: {(end_time - start_time)})')
        return train_cf_array, param_closest_distance

    def construct_graph(self, data, model, feat_values, type):
        """
        Constructs the graph and the required parameters to run Fijuice several lagrange values
        """
        feat_possible_values = self.get_feat_possible_values(data, model)
        print(f'Extracted all possible feature value permutations from training CF. Getting Graph Nodes...')
        graph_nodes = self.get_graph_nodes(data, model, feat_possible_values, type)
        all_nodes = np.concatenate([self.train_cf, graph_nodes], axis=0)
        print(f'Obtained all possible nodes in the graph: {len(all_nodes)}. Calculating costs...')
        C = self.get_all_costs_weights(data, type, all_nodes)
        print(f'Obtained all costs in the graph')
        F = self.get_all_feasibility(data, all_nodes)
        print(f'Obtained all feasibility in the graph')
        rho = self.get_all_likelihood(data, all_nodes, dist=type)
        print(f'Obtained all Likelihood parameter')
        eta = self.get_all_effectiveness(data, feat_values, all_nodes)
        print(f'Obtained all effectiveness parameter')
        return feat_possible_values, all_nodes, C, F, rho, eta
    
    def verify_prediction_feasibility(self, data, model, instance, train_cf, values, i):
        """
        Verifies whether the values are as close as possible to achieve counterfactual state and if the feature can be changed
        """
        instance_feat_val = instance[i]
        if isinstance(instance_feat_val, np.ndarray):
            values_minus_feat_val = np.sum(np.abs(values - instance_feat_val), axis=1)
        else:
            values_minus_feat_val = values - instance_feat_val
        zip_values_difference = list(zip(values, values_minus_feat_val))
        zip_values_difference.sort(key=lambda x: abs(x[1]))
        if isinstance(instance[i], np.ndarray):
            close_cf_values = [list(instance[i])]
        else:
            close_cf_values = [instance[i]]
        v = copy.deepcopy(instance)
        for tup in zip_values_difference:
            value = tup[0]
            v[i] = value
            if verify_feasibility(instance, v, data) and value not in close_cf_values:
                close_cf_values.extend([value])
                if model.model.predict(v.reshape(1, -1)) != self.ioi_label:
                    break
        return close_cf_values

    def get_feat_possible_values(self, data, model, obj=None, points=None):
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
        feat_possible_values_all = {}
        for instance_idx in range(len(sensitive_feature_instances)):
            instance = sensitive_feature_instances[instance_idx]
            cf_feat_possible_values = {}
            for k in range(len(points)):
                train_cf_k = points[k]
                v = train_cf_k - instance 
                nonzero_index = list(np.nonzero(v)[0])
                feat_checked = []
                feat_possible_values = []
                for i in range(len(instance)):
                    if i not in feat_checked:
                        feat_i = data.processed_features[i]
                        if feat_i in data.bin_enc_cols:
                            if i in nonzero_index:
                                value = [train_cf_k[i], instance[i]]
                                value = self.verify_prediction_feasibility(data, model, instance, train_cf_k, value, i)
                            else:
                                value = [train_cf_k[i]]
                            feat_checked.extend([i])
                        elif feat_i in data.cat_enc_cols:
                            idx_cat_i = data.idx_cat_cols_dict[feat_i[:-4]]
                            nn_cat_idx = list(train_cf_k[idx_cat_i])
                            if any(item in idx_cat_i for item in nonzero_index):
                                ioi_cat_idx = list(instance[idx_cat_i])
                                value = [nn_cat_idx, ioi_cat_idx]
                                value = self.verify_prediction_feasibility(data, model, instance, train_cf_k, value, idx_cat_i)
                            else:
                                value = [nn_cat_idx]
                            feat_checked.extend(idx_cat_i)
                        elif feat_i in data.ordinal:
                            if i in nonzero_index:
                                values_i = list(data.processed_feat_dist[feat_i].keys())
                                max_val_i, min_val_i = max(instance[i], train_cf_k[i]), min(instance[i], train_cf_k[i])
                                value = [j for j in values_i if j <= max_val_i and j >= min_val_i]
                                value = self.verify_prediction_feasibility(data, model, instance, train_cf_k, value, i)
                            else:
                                value = [train_cf_k[i]]
                            feat_checked.extend([i])
                        elif feat_i in data.continuous:
                            if i in nonzero_index:
                                max_val_i, min_val_i = max(instance[i], train_cf_k[i]), min(instance[i], train_cf_k[i])
                                value = self.continuous_feat_values(i, min_val_i, max_val_i, data)
                                value = self.verify_prediction_feasibility(data, model, instance, train_cf_k, value, i)
                            else:
                                value = [train_cf_k[i]]
                            feat_checked.extend([i])
                        feat_possible_values.append(value)
                cf_feat_possible_values[k] = feat_possible_values
            feat_possible_values_all[instance_idx] = cf_feat_possible_values
        return feat_possible_values_all

    # def get_graph_nodes(self, data, model, feat_possible_values, type):
    #     """
    #     Generator that contains all the nodes located in the space between the training CFs and the normal_ioi (all possible, CF-labeled nodes)
    #     """
    #     start_time = time.time()
    #     graph_nodes = []
    #     for k in range(len(self.train_cf)):
    #         for instance_idx in range(len(self.sensitive_feature_instances)):
    #             feat_possible_values_k = feat_possible_values[instance_idx][k]
    #             permutations = product(*feat_possible_values_k)
    #             instance = self.sensitive_feature_instances[instance_idx]
    #             for i in permutations:
    #                 perm_i = make_array(i)
    #                 if verify_feasibility(instance, perm_i, data):
    #                     if model.model.predict(perm_i.reshape(1, -1)) != self.ioi_label:
    #                         if not any(np.array_equal(perm_i, x) for x in self.train_cf): #and not any(np.array_equal(perm_i, x) for x in graph_nodes)
    #                             if distance_calculation(instance, perm_i, kwargs={'dat':data, 'type':type}) < self.min_closest_distance:
    #                                 graph_nodes.append(perm_i)
    #             print(f'Graph: instance {instance_idx+1}/{len(self.sensitive_feature_instances)}, Train CF {k+1}/{len(self.train_cf)}. Nodes size: {len(graph_nodes)}')
    #     end_time = time.time()
    #     print(f'Total time (s): {end_time - start_time}')
    #     return graph_nodes

    def get_graph_nodes(self, data, model, feat_possible_values, type):
        """
        Generator that contains all the nodes located in the space between the training CFs and the normal_ioi (all possible, CF-labeled nodes)
        """
        graph_nodes = Parallel(n_jobs=number_cores, verbose=10, prefer='processes')(delayed(get_graph_nodes_parallel)(data,
                                                                              model,
                                                                              self.sensitive_feature_instances,
                                                                              self.train_cf,
                                                                              self.min_closest_distance,
                                                                              feat_possible_values,
                                                                              k,
                                                                              instance_idx,
                                                                              self.ioi_label,
                                                                              type
                                                                              ) for k in range(len(self.train_cf)) for instance_idx in range(len(self.sensitive_feature_instances)) 
                                            )
        graph_nodes_flat_list = list(chain.from_iterable(graph_nodes))
        graph_nodes_array = np.vstack(graph_nodes_flat_list)
        graph_nodes_array_unique = np.unique(graph_nodes_array, axis=0)
        return graph_nodes_array_unique

    def get_all_costs_weights(self, data, type, all_nodes):
        """
        Method that outputs the cost parameters required for optimization
        """
        C = {}
        print(f'Starting pairwise distances...')
        start_time = time.time()
        # distance_mat = pairwise_distances(self.sensitive_feature_instances, all_nodes, metric=distance_calculation, kwargs={'dat':data, 'type':type})
        results_list = Parallel(n_jobs=number_cores, verbose=10, prefer='processes')(delayed(get_all_costs_weights_parallel)(data,
                                                                                                                             self.feat,
                                                                                                                             self.sensitive_feature_instances,
                                                                                                                             all_nodes,
                                                                                                                             self.instance_idx_to_original_idx_dict,
                                                                                                                             self.sensitive_group_idx_feat_value_dict,
                                                                                                                             instance_idx,
                                                                                                                             k,
                                                                                                                             type
                                                                                                                             ) for k in range(1, len(all_nodes) + 1) for instance_idx in range(1, len(self.sensitive_feature_instances) + 1)
                                                                                    )
        for instance_idx, k, distance in results_list:
            C[instance_idx, k] = distance
        end_time = time.time()
        print(f'Total time (s): {(end_time - start_time)}')
        # print(f'Graph Costs: instance {instance_idx}/{len(self.sensitive_feature_instances)}, Train CF {k}/{len(all_nodes)}')
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
        TO BE FIXED
        TO BE FIXED
        TO BE FIXED
        TO BE FIXED: CALCULATE BASED ON FEASIBILITY, NEED TO CHECK WHICH SENSITIVE GROUP DOES THE NODE BELONG TO AND CALCULATE THE LENGTH OF THE SENS. GROUP
        """
        eta = {}
        for feat_value in feat_values:
            feat_value_idx = self.sensitive_group_dict[feat_value]
            len_sensitive_group = len(feat_value_idx)
            for k in range(1, len(all_nodes) + 1):
                sum_eta = 0
                node_k = all_nodes[k - 1]
                for instance_idx in feat_value_idx:
                    instance = data.transformed_false_undesired_test_df.loc[instance_idx].values
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
            

            
            

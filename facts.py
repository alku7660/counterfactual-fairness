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
import multiprocessing
from functools import partial

"""
This method is based on:
Kavouras, L., Tsopelas, K., Giannopoulos, G., Sacharidis, D., Psaroudaki, E., Theologitis, N., ... & Emiris, I. (2023). Fairness Aware Counterfactuals for Subgroups. arXiv preprint arXiv:2306.14978.
"""

class FACTS:

    def __init__(self, counterfactual) -> None:
        data = counterfactual.data
        model = counterfactual.model
        self.support_th = counterfactual.support_th
        self.cluster = counterfactual.cluster
        self.model = model
        self.discretized_train_df = data.discretized_train_df
        self.train_target = data.train_target
        self.undesired_class = data.undesired_class
        self.undesired_discretized_train_df, self.desired_discretized_train_df = self.get_undesired_desired_discretized_train_df()
        self.discretized_test_df = data.discretized_test_df
        self.transformed_test_df = data.transformed_test_df
        self.transformed_test_np = data.transformed_test_np
        self.test_target = data.test_target
        self.undesired_class = data.undesired_class
        self.protected_groups = data.feat_protected
        start_time = time.time()
        self.sensitive_groups = self.get_sensitive_groups()
        self.fpgrowth_df = self.get_fpgrowth_df(data)
        self.fpgrowth_per_feat = self.get_common_fpgrowth_per_sensitive_feature()
        self.actions_fpgrowth_set = self.get_actions_fpgrowth_set()
        self.subgroup_same_cost_actions = self.get_same_cost_actions_per_subgroup(data)
        self.effectiveness_df = self.estimate_effectiveness_per_action_per_sensitive_group(data, model)
        print("G")
        self.best_effectiveness_df = self.select_best_action_per_subgroup()
        print("H")
        self.normal_x_cf, self.actions_x = self.get_cfs_all_fn_instances(data)
        end_time = time.time()
        self.run_time = end_time - start_time

    def get_undesired_desired_discretized_train_df(self):
        """
        Obtains the undesired class and desired class training dataset
        """
        return self.discretized_train_df[self.train_target == self.undesired_class], self.discretized_train_df[self.train_target == 1 - self.undesired_class]

    def get_fpgrowth_df(self, data):
        """
        Obtains the fpgrowth conjunction predicates from the frequent itemsets from the fpgrowth algorithm, as explained in:
        Kavouras, L., Tsopelas, K., Giannopoulos, G., Sacharidis, D., Psaroudaki, E., Theologitis, N., ... & Emiris, I. (2023). Fairness Aware Counterfactuals for Subgroups. arXiv preprint arXiv:2306.14978.
        """
        frequent_subgroups_per_sensitive_group = {}
        for sensitive_feat in self.protected_groups.keys():
            sensitive_groups_dict = self.protected_groups[sensitive_feat]
            sensitive_feat_groups = {}
            for sensitive_group in sensitive_groups_dict.keys():
                if sensitive_feat in data.continuous:
                    column = f'{sensitive_feat}_{sensitive_group}'
                else:
                    column = f'{sensitive_feat}_{int(sensitive_group)}'
                instances_sensitive_group = self.undesired_discretized_train_df[self.undesired_discretized_train_df[column] == 1]
                instances_sensitive_group = instances_sensitive_group.drop(column, axis=1)
                fpgrowth_sensitive_group_df = fpgrowth(instances_sensitive_group, min_support=self.support_th, use_colnames=True)
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
            list_subgroups = value_counts_sensitive_feature_subgroups[value_counts_sensitive_feature_subgroups == len(sensitive_subgroups_dict.keys())].index.to_list()
            common_sensitive_groups_feat = pd.Series(list_subgroups)
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
        itemsets = [i for i in itemsets if len(i) > 0]
        itemsets = pd.Series(itemsets)
        return itemsets
    
    def turn_subgroup_df_into_feat_value_list(self, subgroup_df):
        """
        Turns subgroup_df into a list of tuples containing the feature name and the value of the actions
        """
        list_features_subgroups, list_values_subgroups = list(), list()
        for row in subgroup_df:
            row = list(row)
            row.sort()
            if len(row) == 1:
                feat_split, val_split = [row[0].split('_')[0]], [row[0].split('_')[1]]
            else:
                feat_split, val_split = [i.split('_')[0] for i in row], [i.split('_')[1] for i in row]
            list_features_subgroups.append(feat_split)
            list_values_subgroups.append(val_split)
        return list_features_subgroups, list_values_subgroups

    def get_actions_fpgrowth_set(self):
        """
        Obtains actions from the unaffected training set
        """
        filtered_fpgrowth_actions_df1 = pd.DataFrame()
        filtered_fpgrowth_actions_list1 = list()
        fpgrowth_actions_df = fpgrowth(self.desired_discretized_train_df, min_support=self.support_th, use_colnames=True)
        fpgrowth_actions_srs = self.remove_sensitive_feature_itemsets(fpgrowth_actions_df['itemsets'])
        print(f'Length fpgrowth_actions_srs {len(fpgrowth_actions_srs)}')
        count = 0
        for sensitive_feat in self.fpgrowth_per_feat.keys():
            common_frequent_subgroups_per_sensitive_feat_df = self.fpgrowth_per_feat[sensitive_feat]
            list_features_subgroups, list_values_subgroups = self.turn_subgroup_df_into_feat_value_list(common_frequent_subgroups_per_sensitive_feat_df)
            print(f'Length {sensitive_feat} subgroups: {len(list_features_subgroups)}')
            for action_list in fpgrowth_actions_srs:
                action_list.sort()
                action = action_list[0] if len(action_list) == 1 else action_list
                action_feat = [action.split('_')[0]] if isinstance(action,str) else [i.split('_')[0] for i in action]
                action_value = [action.split('_')[1]] if isinstance(action,str) else [i.split('_')[1] for i in action]
                found_equal_feat_different_value = False
                for idx in range(len(list_features_subgroups)):
                    subgroup_feat, subgroup_val = list_features_subgroups[idx], list_values_subgroups[idx]
                    for ind_action, feat_action in enumerate(action_feat):
                        for ind_subgroup, feat_subgroup in enumerate(subgroup_feat):
                            if feat_action == feat_subgroup:
                                if int(float(action_value[ind_action])) != int(float(subgroup_val[ind_subgroup])):
                                    found_equal_feat_different_value = True
                                    break
                        if found_equal_feat_different_value:
                            break
                    if found_equal_feat_different_value:
                        if action_list not in filtered_fpgrowth_actions_list1:
                            filtered_fpgrowth_actions_list1.append(action_list)
                            count += 1
        filtered_fpgrowth_actions_df1 = pd.Series(filtered_fpgrowth_actions_list1)
        print(f'Length: {len(filtered_fpgrowth_actions_df1)}')
        return filtered_fpgrowth_actions_df1
    
    def get_instances_idx_belonging_to_subgroup(self, subgroup):
        """
        Obtains the instances indices in the false negatives belonging to a given subgroup
        """
        subgroup_instances_idx = []
        for c_idx in range(1, len(self.cluster.filtered_clusters_list) + 1):
            cluster_instances_list = self.cluster.filtered_clusters_list[c_idx - 1]
            for instance_idx in cluster_instances_list:
                instance = self.discretized_test_df.loc[instance_idx].to_frame().T
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
        columns_list = [col for col in self.discretized_test_df.columns if col.split('_')[0] in feat]
        return columns_list

    def get_x_discretized_prime_from_discretized_x(self, action, x):
        """
        Transforms x to x_prime using the action given
        """
        x_prime = copy.deepcopy(x)
        action_feat = [action.split('_')[0]] if isinstance(action,str) else [i.split('_')[0] for i in action]
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
        for cont_feat in data.continuous:
            action_feat = [action.split('_')[0]] if isinstance(action,str) else [i.split('_') for i in action]
            if cont_feat not in action_feat:
                x_prime_transformed[cont_feat] = x_transformed[cont_feat].values
        return x_prime_transformed

    def verify_same_cost_subgroup_action(self, subgroup_instances, action, data):
        """
        Verifies the same cost for all instances in a subgroup for the given action
        """
        cost_instances = []
        for _, x in subgroup_instances.iterrows():
            x = x.to_frame().T
            x_transformed = data.transformed_test_df.loc[x.index,:]
            x_prime = self.get_x_discretized_prime_from_discretized_x(action, x)
            x_prime_normal = self.adjust_continuous_feat_normal_x_prime(x_prime, x_transformed, action, data)
            cost = distance_calculation(np.array(x_transformed)[0], np.array(x_prime_normal).flatten(), kwargs={'dat':data, 'type':'L1_L0'})
            cost_instances.append(cost)
        return len(set(cost_instances)) == 1

    def get_same_cost_actions_per_subgroup_row(self, subgroups, data, sensitive_feat, subgroup, counter):
        subgroup_instances_idx = self.get_instances_idx_belonging_to_subgroup(subgroup)
        subgroup_instances = self.get_instances_with_idx(subgroup_instances_idx)
        same_cost_actions_list = []
        for action in self.actions_fpgrowth_set:
            if self.verify_same_cost_subgroup_action(subgroup_instances, action, data):
                same_cost_actions_list.append(action)
        
        print(f'Analyzing sensitive feature: {sensitive_feat}, Subgroup: {subgroup}. Total subgroups analyzed: {counter}/{len(subgroups)}')
        return same_cost_actions_list, subgroup

    def get_same_cost_actions_per_subgroup(self, data):
        """
        Obtains the subset of actions that have the same cost for all individuals in each subgroup, and that mention at least 1 feature mentioned and at least 1 different feature value.
        """
        action_per_subgroup_dict = {}
        for sensitive_feat in self.protected_groups.keys():
            subgroups = list(self.fpgrowth_per_feat[sensitive_feat])
            ins_idx = zip(subgroups, range(len(subgroups)))

            pool = multiprocessing.Pool(processes=40) 
            func = partial(self.get_same_cost_actions_per_subgroup_row, subgroups, data, sensitive_feat)
            outputs = pool.starmap(func, ins_idx)
            pool.close()
            pool.join()
            for i in outputs:
                 action_per_subgroup_dict[i[1]] = i[0]

            # for subgroup in subgroups:
                # action_per_subgroup_dict[subgroup] = same_cost_actions_list
        return action_per_subgroup_dict


    def get_recourses_for_fn_one_instance(self, data, model, len_ins, x_fn_idx, idx):
        start_time = time.time()
        x = data.discretized_test_df.loc[x_fn_idx,:].to_frame().T
        # print(type(x))
        recourse_set = self.extract_recourses_x(x)
        # 99% of bottleneck comes from here
        results_x = self.results_recourse_rules_x(recourse_set, x, data, model)
        
        end_time = time.time()
        print(f'Dataset: {data.name}. Instance {x_fn_idx} ({idx}/{len_ins}) done (time: {np.round(end_time - start_time, 2)} s)')
        return results_x
    
    def get_recourses_for_fn_instances(self, data, model):
        """
        Obtains all the best recourses for all FN instances
        """
        set_instances = self.fn_instances.index
        ins_idx = zip(set_instances, range(len(set_instances)))

        pool = multiprocessing.Pool(processes=30) 
        func = partial(self.get_recourses_for_fn_one_instance, data, model, len(set_instances))
        outputs = pool.starmap(func, ins_idx)
        pool.close()
        pool.join()
        for i in outputs:
            self.add_results(i)


    def calculate_action_effectiveness(self, subgroup, sensitive_group, action, data, model):
        """
        Given a subgroup, sensitive group, and an action, calculates the effectiveness according to Kavouras, L., Tsopelas, K., Giannopoulos, G., Sacharidis, D., Psaroudaki, E., Theologitis, N., ... & Emiris, I. (2023). Fairness Aware Counterfactuals for Subgroups. arXiv preprint arXiv:2306.14978.
        """
        subgroup_instances_idx = self.get_instances_idx_belonging_to_subgroup(subgroup)
        subgroup_instances_df = self.get_instances_with_idx(subgroup_instances_idx)
        subgroup_instances_sensitive_group_df = subgroup_instances_df.loc[subgroup_instances_df[sensitive_group] == 1]
        effective = 0
        for _, x in subgroup_instances_sensitive_group_df.iterrows():
            x = x.to_frame().T
            normal_x = self.transformed_test_df.loc[x.index,:]
            x_prime = self.get_x_discretized_prime_from_discretized_x(action, x)
            x_prime_normal = self.adjust_continuous_feat_normal_x_prime(x_prime, normal_x, action, data)
            x_pred = model.model.predict(normal_x.values)
            x_prime_pred = model.model.predict(x_prime_normal.values)
            if x_pred == data.undesired_class and x_prime_pred != data.undesired_class:
                effective += 1
            else:
                effective += 0
        if len(subgroup_instances_sensitive_group_df) > 0:
            effectiveness = effective/len(subgroup_instances_sensitive_group_df)
        else:
            effectiveness = 0
        return effectiveness

    def estimate_effectiveness_per_action_per_sensitive_group_row(self, data, model, subgroup, count):
        """
        Estimates the effectiveness of each of the actions for each of the subgroups they apply to. Effectiveness here is simply whether they change the label or not (feasibility is not considered)
        """
        effectiveness_df = pd.DataFrame(columns=cols)

        cols = ['subgroup','sensitive_group','action','effectiveness']
        actions_list = self.subgroup_same_cost_actions[subgroup]
        for sensitive_group in self.sensitive_groups:
            for action in actions_list:
                effectiveness = self.calculate_action_effectiveness(subgroup, sensitive_group, action, data, model)
                effectiveness_row = pd.DataFrame(data=[[subgroup, sensitive_group, action, effectiveness]], index = [count], columns=cols)
                effectiveness_df = pd.concat((effectiveness_df, effectiveness_row))
        print(f'Estimated effectiveness of subgroup actions in: {subgroup}. Total subgroups analyzed for actions effectiveness: {count}/{len(self.subgroup_same_cost_actions.keys())}')
        
        return effectiveness_df
    
        return effectiveness_df
    
    def estimate_effectiveness_per_action_per_sensitive_group(self, data, model):
        """
        Estimates the effectiveness of each of the actions for each of the subgroups they apply to. Effectiveness here is simply whether they change the label or not (feasibility is not considered)
        """
        cols = ['subgroup','sensitive_group','action','effectiveness']
        effectiveness_df = pd.DataFrame(columns=cols)
        count = 0
        subgroups = self.subgroup_same_cost_actions.keys()
        ins_idx = zip(subgroups, range(len(subgroups)))

        pool = multiprocessing.Pool(processes=40) 
        func = partial(self.estimate_effectiveness_per_action_per_sensitive_group_row, data, model)
        outputs = pool.starmap(func, ins_idx)
        pool.close()
        pool.join()
        effectiveness_df = pd.concat(outputs)

        # for subgroup in self.subgroup_same_cost_actions.keys():
        #     count += 1
        #     actions_list = self.subgroup_same_cost_actions[subgroup]
        #     for sensitive_group in self.sensitive_groups:
        #         for action in actions_list:
        #             effectiveness = self.calculate_action_effectiveness(subgroup, sensitive_group, action, data, model)
        #             effectiveness_row = pd.DataFrame(data=[[subgroup, sensitive_group, action, effectiveness]], index = [count], columns=cols)
        #             effectiveness_df = pd.concat((effectiveness_df, effectiveness_row))
        #     print(f'Estimated effectiveness of subgroup actions in: {subgroup}. Total subgroups analyzed for actions effectiveness: {count}/{len(self.subgroup_same_cost_actions.keys())}')
        effectiveness_df.sort_values('effectiveness', ascending=False)
        return effectiveness_df
    
    def estimate_effectiveness_per_action_per_sensitive_group_old(self, data, model):
        """
        Estimates the effectiveness of each of the actions for each of the subgroups they apply to. Effectiveness here is simply whether they change the label or not (feasibility is not considered)
        """
        cols = ['subgroup','sensitive_group','action','effectiveness']
        effectiveness_df = pd.DataFrame(columns=cols)
        count = 0
        for subgroup in self.subgroup_same_cost_actions.keys():
            count += 1
            actions_list = self.subgroup_same_cost_actions[subgroup]
            for sensitive_group in self.sensitive_groups:
                for action in actions_list:
                    effectiveness = self.calculate_action_effectiveness(subgroup, sensitive_group, action, data, model)
                    effectiveness_row = pd.DataFrame(data=[[subgroup, sensitive_group, action, effectiveness]], index = [count], columns=cols)
                    effectiveness_df = pd.concat((effectiveness_df, effectiveness_row))
            print(f'Estimated effectiveness of subgroup actions in: {subgroup}. Total subgroups analyzed for actions effectiveness: {count}/{len(self.subgroup_same_cost_actions.keys())}')
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
            subgroup_effectiveness_df.sort_values('effectiveness',ascending=False)
            best_effectiveness_for_subgroup = subgroup_effectiveness_df.iloc[0,:].to_frame().T
            best_effectiveness_action_df = pd.concat((best_effectiveness_action_df, best_effectiveness_for_subgroup))
        return best_effectiveness_action_df

    def get_cfs_all_fn_instances(self, data):
        """
        Gets the counterfactuals, based on the best actions, for the given subgroups and for each instance
        """
        cfs_dict, action_dict = dict(), dict()
        for _, row in self.best_effectiveness_df.iterrows():
            row = row.to_frame().T
            subgroup, action = row['subgroup'], list(row['action'])[0]
            subgroup = list(subgroup.iloc[0])
            subgroup_instances_idx = self.get_instances_idx_belonging_to_subgroup(subgroup)
            subgroup_instances_df = self.get_instances_with_idx(subgroup_instances_idx)
            for _, subgroup_instance in subgroup_instances_df.iterrows():
                subgroup_instance = subgroup_instance.to_frame().T
                idx = subgroup_instance.index[0]
                x_transformed = self.transformed_test_df.loc[subgroup_instance.index,:]
                x_discretized_prime = self.get_x_discretized_prime_from_discretized_x(action, subgroup_instance)
                x_prime_normal = self.adjust_continuous_feat_normal_x_prime(x_discretized_prime, x_transformed, action, data)
                cfs_dict[idx] = x_prime_normal
                action_dict[idx] = action
        return cfs_dict, action_dict

    def get_sensitive_groups(self):
        """
        Obtains a list of sensitive groups
        """
        sensitive_group_list = []
        for sensitive_group in self.protected_groups.keys():
            sensitive_group_list.extend([x for x in self.discretized_train_df.columns if sensitive_group in x])
        return sensitive_group_list
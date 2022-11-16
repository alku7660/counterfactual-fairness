"""
Evaluation algorithms
"""

"""
Imports
"""
import numpy as np
import pandas as pd
from support import euclidean, sort_data_distance
from nn_method import near_neigh
from mo import min_obs
from rt import rf_tweak
from cchvae_cf import cchvae_function
import copy
# from carla.recourse_methods import Face, Dice, GrowingSpheres (REQUIRES ADJUSTMENT / FUTURE WORK)

class Evaluator():
    """
    DESCRIPTION:        Evaluator Class
    
    INPUT:
    data_obj:           Dataset object
    n_feat:             Number of examples to generate synthetically per feature
    method_str:         Name of the method to use to obtain counterfactuals
    """
    def __init__(self, data_obj, n_feat, method_str):
        self.data_name = data_obj.name
        self.method_name = method_str
        self.feat_type = data_obj.feat_type
        self.feat_mutable = data_obj.feat_mutable
        self.feat_cost = data_obj.feat_cost
        self.feat_step = data_obj.feat_step
        self.feat_dir = data_obj.feat_dir
        self.feat_protected = data_obj.feat_protected
        self.binary = data_obj.binary
        self.categorical = data_obj.categorical
        self.numerical = data_obj.numerical
        self.bin_enc, self.bin_enc_cols  = data_obj.bin_enc, data_obj.bin_enc_cols
        self.cat_enc, self.cat_enc_cols  = data_obj.cat_enc, data_obj.cat_enc_cols
        self.scaler = data_obj.scaler
        self.data_cols = data_obj.transformed_cols
        self.raw_data_cols = data_obj.train_df.columns
        self.undesired_class = data_obj.undesired_class
        self.desired_class = 1 - self.undesired_class
        self.n_feat = n_feat
        self.x, self.original_x, self.x_pred, self.x_target, self.x_accuracy = {}, {}, {}, {}, {}
        self.cf, self.original_cf = {}, {} 
        self.group_cf, self.original_group_cf = None, None
        self.cf_proximity, self.cf_feasibility, self.cf_sparsity, self.cf_validity, self.cf_time = {}, {}, {}, {}, {}
        self.group_cf_proximity, self.group_cf_feasibility, self.group_cf_sparsity, self.group_cf_validity = {}, {}, {}, {}, None
        self.x_clusters, self.original_x_clusters = {}, {}
        self.x_clusters_cf, self.original_x_clusters_cf = {}, {}
        self.cluster_cf_proximity = pd.DataFrame(index=self.x.keys(), columns=self.x_clusters.keys())
        self.cluster_cf_feasibility = pd.DataFrame(index=self.x.keys(), columns=self.x_clusters.keys())
        self.cluster_cf_sparsity = pd.DataFrame(index=self.x.keys(), columns=self.x_clusters.keys())
        self.cluster_cf_time = {}

    def search_desired_class_penalize(self, data):
        """
        DESCRIPTION:        Obtains the penalization for the method if no instance of the desired class is obtained as CF

        INPUT:
        data:               Dataset object

        OUTPUT:
        penalize_instance:  The furthest training instance available
        """
        train_desired_class = data.transformed_train_np[data.train_target != self.undesired_class]
        sorted_train_x = sort_data_distance(self.x,train_desired_class,data.train_target[data.train_target != self.undesired_class])
        penalize_instance = sorted_train_x[-1][0]
        return penalize_instance

    def statistical_parity_eval(self, prob, length):
        """
        DESCRIPTION:        Calculates the Statistical Parity measure for each of the protected feature groups given a model

        INPUT:
        prob:               Probability dictionary
        length:             Number of instances belonging to each feature value or group

        OUTPUT:
        stat_parity:        Statistical parity measure
        """
        stat_parity = {}
        for i in self.feat_protected.keys():
            feat_names = prob[i].keys()
            feat_parity = {}
            for j in feat_names:
                j_probability = prob[i][j]
                total_others_length = np.sum([length[i][x] for x in feat_names if x != j])
                others_probability = np.sum([prob[i][x]*length[i][x] for x in feat_names if x != j]) / total_others_length
                feat_parity[j] = j_probability - others_probability
            stat_parity[i] = feat_parity
        return stat_parity

    def statistical_parity_proba(self, data_obj, model_obj):
        """
        DESCRIPTION:        Calculates the probabilities used by the Statistical Parity method

        INPUT:
        data_obj:           Dataset object
        model_obj:          Model object

        OUTPUT:
        prob_dict:          Probability dictionary
        length_dict:        Dictionary with the number of instances belonging to each feature value or group
        """
        prob_dict = {}
        length_dict = {}
        test_data = data_obj.test_df
        test_data_transformed = data_obj.transformed_test_df
        test_data_transformed_index = test_data_transformed.index
        pred = model_obj.sel.predict(test_data_transformed)
        pred_df = pd.DataFrame(data=pred, index=test_data_transformed_index, columns=['prediction'])
        test_data_with_pred = pd.concat((test_data, pred_df),axis=1)
        for i in self.feat_protected.keys():
            feat_i_values = test_data_with_pred[i].unique()
            val_dict = {}
            length_val_dict = {}
            for j in feat_i_values:
                val_name = self.feat_protected[i][np.round(j,2)]
                total_feat_i_val_j = len(test_data_with_pred[test_data_with_pred[i] == j])
                total_feat_i_val_j_desired_pred = len(test_data_with_pred[(test_data_with_pred[i] == j) & (test_data_with_pred['prediction'] == self.desired_class)])
                stat_parity_feat_val = total_feat_i_val_j_desired_pred / total_feat_i_val_j
                val_dict[val_name] = stat_parity_feat_val
                length_val_dict[val_name] = total_feat_i_val_j
            prob_dict[i] = val_dict
            length_dict[i] = length_val_dict
        return prob_dict, length_dict

    def equalized_odds_eval(self, prob, length):
        """
        DESCRIPTION:        Calculates the Equalized Odds measure for each of the protected feature groups given a model

        INPUT:
        prob:               Probability dictionary
        length:             Number of instances belonging to each feature value or group

        OUTPUT:
        eq_odds:            Equalized odds measure
        """
        eq_odds = {}
        for i in self.feat_protected.keys():
            feat_val_names = prob[i].keys()
            feat_odds = {}
            for j in feat_val_names:
                j_probability_target_0, j_probability_target_1 = prob[i][j][0], prob[i][j][1]
                total_others_target_0_length = np.sum([length[i][x][0] for x in feat_val_names if x != j])
                total_others_target_1_length = np.sum([length[i][x][1] for x in feat_val_names if x != j])
                others_probability_target_0 = np.sum([prob[i][x][0]*length[i][x][0] for x in feat_val_names if x != j]) / total_others_target_0_length
                others_probability_target_1 = np.sum([prob[i][x][1]*length[i][x][1] for x in feat_val_names if x != j]) / total_others_target_1_length
                feat_odds[j] = np.abs(j_probability_target_0 - others_probability_target_0) + np.abs(j_probability_target_1 - others_probability_target_1)
            eq_odds[i] = feat_odds
        return eq_odds

    def equalized_odds_proba(self, data_obj, model_obj):
        """
        DESCRIPTION:        Calculates the probabilities used by the Equalized Odds method

        INPUT:
        data_obj:           Dataset object
        model_obj:          Model object

        OUTPUT:
        prob_dict:          Probability dictionary
        length_dict:        Dictionary with the number of instances belonging to each feature value or group
        """
        prob_dict = {}
        length_dict = {}
        test_data = data_obj.test_df
        test_data_transformed = data_obj.transformed_test_df
        test_data_transformed_index = test_data_transformed.index
        test_target = data_obj.test_target
        pred = model_obj.sel.predict(test_data_transformed)
        pred_df = pd.DataFrame(data=pred, index=test_data_transformed_index, columns=['prediction'])
        target_df = pd.DataFrame(data=test_target, index=test_data_transformed_index, columns=['target'])
        test_data_with_pred_target = pd.concat((test_data, pred_df, target_df),axis=1)
        for i in self.feat_protected.keys():
            feat_i_values = test_data_with_pred_target[i].unique()
            feat_dict = {}
            length_val_dict = {}
            for j in feat_i_values:
                val_name = self.feat_protected[i][np.round(j,2)]
                target_dict = {}
                length_target_dict = {}
                for k in [0,1]:
                    data_feat_i_val_j_ground_k = test_data_with_pred_target[(test_data_with_pred_target[i] == j) & (test_data_with_pred_target['target'] == k)]
                    data_feat_i_val_j_ground_k_desired_pred = data_feat_i_val_j_ground_k[data_feat_i_val_j_ground_k['prediction'] == self.desired_class]
                    total_feat_i_val_j_ground_k = len(data_feat_i_val_j_ground_k)
                    total_feat_i_val_j_ground_k_desired_pred = len(data_feat_i_val_j_ground_k_desired_pred)
                    prob_val = total_feat_i_val_j_ground_k_desired_pred / total_feat_i_val_j_ground_k
                    target_dict[k] = prob_val
                    length_target_dict[k] = total_feat_i_val_j_ground_k
                feat_dict[val_name] = target_dict
                length_val_dict[val_name] = length_target_dict
            prob_dict[i] = feat_dict
            length_dict[i] = length_val_dict
        return prob_dict, length_dict

    def add_fairness_measures(self, data_obj, model_obj):
        """
        DESCRIPTION:        Adds the fairness metrics Statistical Parity and Equalized Odds

        INPUT:
        data_obj:           Dataset object
        model_obj:          Model object

        OUTPUT: (None: stored as class attributes)
        prob_dict:          Probability dictionary
        length_dict:        Dictionary with the number of instances belonging to each feature value or group
        """
        stat_proba, stat_length = self.statistical_parity_proba(data_obj, model_obj)
        odds_proba, odds_length = self.equalized_odds_proba(data_obj, model_obj)
        self.stat_parity = self.statistical_parity_eval(stat_proba, stat_length)
        self.eq_odds = self.equalized_odds_eval(odds_proba, odds_length)

    def add_fnr_data(self, desired_ground_truth_test_df, false_undesired_test_df, transformed_false_undesired_test_df):
        """
        DESCRIPTION:                            Adds the desired ground truth test DataFrame and the false negative test DataFrame
        
        INPUT:
        desired_ground_truth_test_df:           Desired ground truth DataFrame
        false_undesired_test_df:                False negative test DataFrame
        transformed_false_undesired_test_df:    Transformed false negative test DataFrame

        OUTPUT: (None: stored as class attributes)
        """
        self.desired_ground_truth_test_df = desired_ground_truth_test_df
        self.false_undesired_test_df = false_undesired_test_df
        self.transformed_false_undesired_test_df = transformed_false_undesired_test_df

    def add_specific_x_data(self, idx, x, original_x, x_pred, x_target):
        """
        DESCRIPTION:        Calculates and stores x data found in the Evaluator

        INPUT:
        idx:                Index of the instance x
        x:                  Instance of interest in Numpy array
        original_x:         Instance of interest in original format (before normalization and encoding) in DataFrame
        x_pred:             Predicted label of the instance of interest
        x_target:           Ground truth label of the instance of interest

        OUTPUT: (None: stored as class attributes)
        """
        self.x[idx] = pd.DataFrame(data=[x], index=[idx], columns=self.data_cols)
        self.original_x[idx] = original_x
        self.x_pred[idx] = x_pred[0]
        self.x_target[idx] = x_target
        self.x_accuracy[idx] = self.accuracy(idx)

    def inverse_transform_original(self, instance):
        """
        DESCRIPTION:            Transforms an instance to the original features
        
        INPUT:
        instance:               Instance of interest

        OUTPUT:
        original_instance_df:   Instance of interest in the original feature format
        """
        instance_index = instance.index
        original_instance_df = pd.DataFrame(index=instance_index)
        if len(self.bin_enc_cols) > 0:
            instance_bin = self.bin_enc.inverse_transform(instance[self.bin_enc_cols])
            instance_bin_pd = pd.DataFrame(data=instance_bin,index=instance_index,columns=self.binary)
            original_instance_df = pd.concat((original_instance_df,instance_bin_pd),axis=1)
        if len(self.cat_enc_cols) > 0:
            instance_cat = self.cat_enc.inverse_transform(instance[self.cat_enc_cols])
            instance_cat_pd = pd.DataFrame(data=instance_cat,index=instance_index,columns=self.categorical)
            original_instance_df = pd.concat((original_instance_df,instance_cat_pd),axis=1)
        if len(self.numerical) > 0:
            instance_num = self.scaler.inverse_transform(instance[self.numerical])
            instance_num_pd = pd.DataFrame(data=instance_num,index=instance_index,columns=self.numerical)
            original_instance_df = pd.concat((original_instance_df,instance_num_pd),axis=1)
        return original_instance_df

    def add_specific_cf_data(self, idx, data_obj, cf, cf_time):
        """
        DESCRIPTION:        Calculates and stores a cf method result and performance metrics into the Pandas DataFrame found in the Evaluator

        INPUT:
        idx:                Index of the instance of interest
        data_obj:           Dataset object
        cf:                 Counterfactual instance obtained
        cf_time:            Run time for the counterfactual method used
        """
        if cf is not None and not np.isnan(np.sum(cf)):
            if isinstance(cf, pd.DataFrame):
                self.cf[idx] = cf
            elif isinstance(cf, pd.Series):
                cf_np = cf.to_numpy()
                self.cf[idx] = pd.DataFrame(data=[cf_np], index=[idx], columns=data_obj.transformed_cols) 
            else:
                self.cf[idx] = pd.DataFrame(data = [cf], index = [idx], columns = data_obj.transformed_cols)
        else:
            penalize_instance = self.search_desired_class_penalize(data_obj)
            self.cf[idx] = pd.DataFrame(data=[penalize_instance], index=[idx], columns=data_obj.transformed_cols)
        self.cf_validity[idx] = True
        self.original_cf[idx] = self.inverse_transform_original(self.cf[idx])
        self.proximity(idx)
        self.feasibility(idx)
        self.sparsity(data_obj, idx)
        self.cf_time[idx] = cf_time

    def accuracy(self, idx):
        """
        DESCRIPTION:        Estimates accuracy between the target and the predicted label of x

        INPUT:
        idx:                Index of the instance of interest

        OUTPUT:
        acc:                Accuracy (agreement) between the predicted and ground truth labels 
        """
        acc = self.x_target[idx] == self.x_pred[idx]
        return acc

    def proximity(self, idx, group=False, cluster_str=None):
        """
        DESCRIPTION:        Calculates the distance between the instance of interest and the counterfactual given

        INPUT:
        idx:                Index of the instance of interest
        group:              Whether to calculate proximity w.r.t. individual counterfactual or the group counterfactual
        cluster_str:        Cluster counterfactual name

        OUTPUT: (None: stored as class attributes)
        """
        if group and cluster_str is None:
            cf = self.group_cf
        elif not group and cluster_str is not None:
            cf = self.add_clusters_cf[cluster_str] 
        else:
            cf = self.cf[idx]
        if (self.x[idx].columns == cf.columns).all():
            distance = np.round_(euclidean(self.x[idx].to_numpy(), cf.to_numpy()), 3)
        else:
            instance_x_copy = copy.deepcopy(self.x[idx])
            instance_x_copy = instance_x_copy[cf.columns]
            distance = np.round_(euclidean(instance_x_copy.to_numpy(), cf.to_numpy()), 3)
        if group and cluster_str is None:
            self.group_cf_proximity[idx] = distance
        elif not group and cluster_str is not None:
            self.cluster_cf_proximity.loc[idx, cluster_str] = distance
        else:
            self.cf_proximity[idx] = distance

    def feasibility(self, idx, group=False, cluster_str=None):
        """
        DESCRIPTION:        Indicates whether cf is a feasible counterfactual with respect to x and the feature mutability

        INPUT:
        idx:                Index of the instance of interest
        group:              Whether to calculate feasibility w.r.t. individual counterfactual or the group counterfactual
        cluster_str:        Cluster counterfactual name

        OUTPUT: (None: stored as class attributes)
        """
        toler = 0.000001
        feasibility = True
        if group and cluster_str is None:
            cf_idx = self.group_cf
        elif not group and cluster_str is not None:
            cf_idx = self.add_clusters_cf[cluster_str] 
        else:
            cf_idx = self.cf[idx]
        x_idx = self.x[idx].to_numpy()[0]
        vector = cf_idx.to_numpy()[0] - x_idx
        for i in range(len(self.feat_type)):
            if self.feat_type[i] == 'bin':
                if not np.isclose(cf_idx.iloc[0,i], [0,1], atol=toler).any():
                    feasibility = False
                    break
            elif self.feat_type[i] == 'num-ord':
                possible_val = np.linspace(0,1,int(1/self.feat_step[i]+1),endpoint=True)
                if not np.isclose(cf_idx.iloc[0,i], possible_val, atol=toler).any():
                    feasibility = False
                    break
            else:
                if cf_idx.iloc[0,i] < 0-toler or cf_idx.iloc[0,i] > 1+toler:
                    feasibility = False
                    break
            if self.feat_dir[i] == 0 and vector[i] != 0:
                feasibility = False
                break
            elif self.feat_dir[i] == 'pos' and vector[i] < 0:
                feasibility = False
                break
            elif self.feat_dir[i] == 'neg' and vector[i] > 0:
                feasibility = False
                break
        if not np.array_equal(x_idx[np.where(self.feat_mutable == 0)], cf_idx[np.where(self.feat_mutable == 0)]):
            feasibility = False
        if group and cluster_str is None:
            self.group_cf_feasibility[idx] = feasibility
        elif not group and cluster_str is not None:
            self.cluster_cf_feasibility.loc[idx, cluster_str] = feasibility
        else:
            self.cf_feasibility[idx] = feasibility

    def sparsity(self, data_obj, idx, group=False, cluster_str=None):
        """
        DESCRIPTION:        Calculates sparsity for a given counterfactual according to x. Sparsity is 1 - the fraction of features changed in the cf. Takes the value of 1 if the number of changed features is 1

        INPUT:
        data_obj:           Dataset object
        idx:                Index of the instance of interest
        group:              Whether to calculate sparsity w.r.t. individual counterfactual or the group counterfactual
        cluster_str:        Cluster counterfactual name

        OUTPUT: (None: stored as class attributes)
        """
        if group and cluster_str is None:
            cf = self.group_cf
        elif not group and cluster_str is not None:
            cf = self.add_clusters_cf[cluster_str] 
        else:
            cf = self.cf[idx]
        cf_idx = cf.to_numpy()[0]
        x_idx = self.x[idx].to_numpy()[0]
        unchanged_features = np.sum(np.equal(x_idx, cf_idx))
        categories_feat_changed = data_obj.feat_cat[np.where(np.equal(x_idx, cf_idx) == False)[0]]
        len_categories_feat_changed_unique = len([i for i in np.unique(categories_feat_changed) if 'cat' in i])
        unchanged_features += len_categories_feat_changed_unique
        n_changed = len(x_idx) - unchanged_features
        if n_changed == 1:
            sparsity = 1.000
        else:
            sparsity = np.round_(1 - n_changed/len(x_idx), 3)
        if group and cluster_str is None:
            self.group_cf_sparsity[idx] = sparsity
        elif not group and cluster_str is not None:
            self.cluster_cf_sparsity.loc[idx, cluster_str] = sparsity
        else:
            self.cf_sparsity[idx] = sparsity

    def validity(self, model_obj):
        """
        DESCRIPTION:            Calculates the validity of the group counterfactual

        INPUT:
        model_obj:              Model object

        OUTPUT: (None: stored as class attributes)
        """
        self.group_cf_validity = model_obj.sel.predict(self.group_cf) != self.undesired_class

    def evaluate_cf_models(self, idx, x_np, x_pred, data_obj, model_obj, epsilon_ft, carla_model, x_original):
        """
        DESCRIPTION:        Evaluates the specific counterfactual method on the isntance of interest

        INPUT:
        idx:                Index of the instance of interest
        x_np:               Instance of interest in Numpy array format
        x_pred:             Predicted label of the instance of interest
        data_obj:           Dataset object
        model_obj:          Model object
        epsilon_ft:         Parameter for the Feature Tweaking counterfactual method
        carla_model:        CARLA framework classifier model
        x_original:         Instance of interest in the original format (For CARLA framework transformation)

        OUTPUT: (None: stored as class attributes)
        """
        if 'mutable' in self.method_name:
            mutability_check = False
        else:
            mutability_check = True
        if 'nn' in self.method_name:
            cf, run_time = near_neigh(x_np ,x_pred, data_obj, mutability_check)
        elif 'mo' in self.method_name:
            cf, run_time = min_obs(x_np, x_pred, data_obj, mutability_check)
        elif 'rt' in self.method_name:
            cf, run_time = rf_tweak(x_np, x_pred, model_obj.rf, data_obj, True, mutability_check)
        elif 'cchvae' in self.method_name:
            cf, run_time = cchvae_function(data_obj, carla_model, x_original)
            
        # WORK IN PROGRESS:
        # elif 'ft' in self.method_name:
        #     cf, run_time = feat_tweak(x_np, model_obj.rf, epsilon_ft)
        # elif 'face' in self.method_name:
        #     cf, run_time = face_function(data_obj, carla_model, x_original)
        # elif 'gs' in self.method_name:
        #     cf, run_time = gs_function(data_obj, carla_model, x_original)
        # elif 'dice' in self.method_name:
        #     cf, run_time = dice_function(data_obj, carla_model, x_original)
        # elif 'juice' in self.method_name:
        #     results = JUICE(x_np, x_pred, data_obj, model_obj.sel, 'proximity', mutability_check)
        #     cf, run_time = results[0], results[4]

        print(f'  {self.method_name} (time (s): {np.round_(run_time, 2)})')
        print(f'---------------------------')
        self.add_specific_cf_data(idx, data_obj, cf, run_time)
    
    def add_group_cf(self):
        """
        DESCRIPTION:            Adds the group counterfactual by finding the mean point of all counterfactuals for the false negative group of test instances

        INPUT: (None)

        OUTPUT: (None: stored as class attributes)
        """
        cf_df = pd.concat(self.cf.values(), axis=0)
        self.group_cf = cf_df.mean(axis=0).to_frame.T
        self.original_group_cf = self.inverse_transform_original(self.group_cf)

    def evaluate_group_cf(self, data_obj, model_obj):
        """
        DESCRIPTION:            Obtains all the performance measures for the group counterfactual

        INPUT:
        data_obj:               Dataset object
        model_obj:              Model object

        OUTPUT: (None: stored as class attributes)
        """
        for idx in self.original_x.keys():
            self.proximity(idx, group=True)
            self.feasibility(idx, group=True)
            self.sparsity(data_obj, idx, group=True)
            self.validity(model_obj)
    
    def add_clusters(self):
        """
        DESCRIPTION:            Adds the clusters for all instances of interest and per sensitive group

        INPUT: (None)

        OUTPUT: (None: stored as class attributes)      
        """
        x_df = pd.concat(self.x.values(), axis=0)
        self.x_clusters['all'] = x_df.mean(axis=0).to_frame.T
        self.original_x_clusters['all'] = self.inverse_transform_original(self.x_cluster_all)
        for feat in self.feat_protected:
            feat_unique_val = x_df[feat].unique()
            for feat_val in feat_unique_val:
                x_df_feat_val = x_df[x_df[feat] == feat_val]
                feat_val_name = self.feat_protected[feat][np.round(feat_val, 2)]
                self.x_clusters[feat_val_name] = x_df_feat_val.mean(axis=0).to_frame.T
                self.original_x_clusters[feat_val_name] = self.inverse_transform_original(self.x_cluster_per_feat[feat_val_name])
    
    def add_cluster_cf_data(self, cluster_str, data_obj, cluster_cf, run_time):
        """
        DESCRIPTION:            Stores the cluster CF and obtains all the performance measures for the cluster counterfactual 
        
        INPUT:
        cluster_str:            Cluster counterfactual name
        data_obj:               Dataset object
        cluster_cf:             Cluster counterfactual
        run_time:               Run time of the cluster counterfactual algorithm

        OUTPUT: (None: stored as class attributes)
        """
        self.x_clusters_cf[cluster_str] = cluster_cf
        self.original_x_clusters_cf[cluster_str] = self.inverse_transform_original(cluster_cf)
        self.cluster_cf_time[cluster_str] = run_time
        for idx in self.x.keys():
            self.proximity(idx, cluster_str=cluster_str)
            self.feasibility(idx, cluster_str=cluster_str)
            self.sparsity(data_obj, idx, cluster_str=cluster_str)

    def add_clusters_cf(self, data_obj, model_obj, carla_model):
        """
        DESCRIPTION:            Find counterfactuals to the clusters found

        INPUT:
        data_obj:               Dataset object
        model_obj:              Model object
        carla_model:            CARLA framework model

        OUTPUT: (None: stored as class attributes)
        """
        clusters_list = self.x_clusters.keys()
        for cluster_str in clusters_list:
            x_cluster = self.x_clusters[cluster_str].to_numpy()
            x_cluster_pred = model_obj.sel.predict(x_cluster.reshape(1, -1))
            x_original = self.original_x_clusters[cluster_str]
            if 'mutable' in self.method_name:
                mutability_check = False
            else:
                mutability_check = True
            if 'nn' in self.method_name:
                cluster_cf, run_time = near_neigh(x_cluster, x_cluster_pred, data_obj, mutability_check)
            elif 'mo' in self.method_name:
                cluster_cf, run_time = min_obs(x_cluster, x_cluster_pred, data_obj, mutability_check)
            elif 'rt' in self.method_name:
                cluster_cf, run_time = rf_tweak(x_cluster, x_cluster_pred, model_obj.rf, data_obj, True, mutability_check)
            elif 'cchvae' in self.method_name:
                cluster_cf, run_time = cchvae_function(data_obj, carla_model, x_original)

            print(f'  {self.method_name} (all cluster cf time (s): {np.round_(run_time, 2)})')
            print(f'---------------------------')
            self.add_cluster_cf_data(cluster_str, data_obj, cluster_cf, run_time)
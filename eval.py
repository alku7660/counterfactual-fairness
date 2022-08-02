"""
Evaluation algorithms
"""

"""
Imports
"""

import numpy as np
import pandas as pd
from support import euclidean, sort_data_distance
from nt import nn
from mo import min_obs
from rt import rf_tweak
from cchvae_cf import cchvae_function
import time
import copy
from carla import DataCatalog, MLModelCatalog
# from carla.recourse_methods import Face, Dice, GrowingSpheres (REQUIRES ADJUSTMENT / FUTURE WORK)
import tensorflow as tf
from carla_adapter import MyOwnDataSet, MyOwnModel

class Evaluator():
    """
    Dataset Class: The class contains the following attributes:
        (0) data_name:                 Dataset name used 
        (1) feat_type:                 Feature type vector from the Dataset object,
        (2) feat_mutable:              Mutability vector from the Dataset object,
        (3) feat_cost:                 Cost vector from the Dataset object,
        (4) feat_step:                 Step size vector from the Dataset object,
        (5) feat_dir:                  Directionality vector from the Dataset object,
        (6) feat_protected:            Set of protected features from the Dataset object,
        (7) binary:                    Name of the binary features in the dataset,
        (8) categorical:               Name of the categorical features in the dataset,
        (9) numerical:                 Name of the numerical features in the dataset,
       (10) oh_bin_enc:                One-hot-encoded binary features,
       (11) oh_bin_enc_cols:           One-hot-encoded binary features names,
       (12) oh_cat_enc:                One-hot-encoded categorical features,
       (13) oh_bin_enc_cols:           One-hot-encoded categorical features names,
       (14) scaler:                    Scaler object,
       (15) data_cols:                 All dataset columns after preprocessing,
       (16) raw_data_cols:             All dataset columns before preprocessing,
       (17) undesired_class:           Dataset undesired class,
       (18) desired_class:             Dataset desired class,
       (19) n_feat:                    Number of instances to generate per feature in the continuous feature space (only used to verify Justification and run Juice algo),
       (20) models:                    List of models names to run ['nn','mo','rt','cchvae'],
       (21) x_columns:                 Instance of Interest information,
       (22) eval_columns:              Evaluation columns names (mainly performance metrics related to counterfactuals),
       (23) all_x_data:                Instance of Interest pandas DataFrame,
       (24) all_cf_data:               Counterfactuals pandas DataFrame.
    """

    def __init__(self,data_obj,n_feat,models):
        self.data_name = data_obj.name
        self.feat_type = data_obj.feat_type
        self.feat_mutable = data_obj.feat_mutable
        self.feat_cost = data_obj.feat_cost
        self.feat_step = data_obj.feat_step
        self.feat_dir = data_obj.feat_dir
        self.feat_protected = data_obj.feat_protected
        self.binary = data_obj.jce_binary
        self.categorical = data_obj.jce_categorical
        self.numerical = data_obj.jce_numerical
        self.oh_bin_enc, self.oh_bin_enc_cols  = data_obj.oh_jce_bin_enc, data_obj.oh_jce_bin_enc_cols
        self.oh_cat_enc, self.oh_cat_enc_cols  = data_obj.oh_jce_cat_enc, data_obj.oh_jce_cat_enc_cols
        self.scaler = data_obj.jce_scaler
        self.data_cols = data_obj.jce_all_cols
        self.raw_data_cols = data_obj.train_pd.columns
        self.undesired_class = data_obj.undesired_class
        self.desired_class = 1 - self.undesired_class
        self.n_feat = n_feat
        self.models = models
        self.x_columns = ['x','normal_x','original_x',
                          'x_pred','x_target','accuracy']
        self.eval_columns = ['cf_method','cf','normal_cf',
                             'original_cf','proximity','feasibility',
                             'sparsity','valid','time']
        self.all_x_data = pd.DataFrame(columns=self.x_columns)
        self.all_cf_data = pd.DataFrame(columns=self.eval_columns)
    
    def search_desired_class_penalize(self, data):
        """
        Method to obtain the penalization for the method if no instance of the desired class is obtained as CF
        """
        train_desired_class = data.jce_train_np[data.train_target != self.undesired_class]
        sorted_train_x = sort_data_distance(self.x,train_desired_class,data.train_target[data.train_target != self.undesired_class])
        penalize_instance = sorted_train_x[-1][0]
        return penalize_instance

    def statistical_parity_eval(self, prob, length):
        """
        Method that calculates the Statistical Parity measure for each of the protected feature groups given a model
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
        Method that calculates the probabilities used by the Statistical Parity method
        """
        prob_dict = {}
        length_dict = {}
        test_data = data_obj.test_pd
        test_data_jce = data_obj.jce_test_pd
        test_data_jce_index = test_data_jce.index
        pred = model_obj.jce_sel.predict(test_data_jce)
        pred_pd = pd.DataFrame(data=pred, index=test_data_jce_index, columns=['prediction'])
        test_data_with_pred = pd.concat((test_data, pred_pd),axis=1)
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
        Method that calculates the Equalized Odds measure for each of the protected feature groups given a model
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
        Method that calculates the probabilities used by the Equalized Odds method
        """
        prob_dict = {}
        length_dict = {}
        test_data = data_obj.test_pd
        test_data_jce = data_obj.jce_test_pd
        test_data_jce_index = test_data_jce.index
        test_target = data_obj.test_target
        pred = model_obj.jce_sel.predict(test_data_jce)
        pred_pd = pd.DataFrame(data=pred, index=test_data_jce_index, columns=['prediction'])
        target_pd = pd.DataFrame(data=test_target, index=test_data_jce_index, columns=['target'])
        test_data_with_pred_target = pd.concat((test_data, pred_pd, target_pd),axis=1)
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

    def add_fairness_measures(self, data_obj, model):
        """
        Method that adds the fairness metrics Statistical Parity and Equalized Odds
        """
        stat_proba, stat_length = self.statistical_parity_proba(data_obj, model)
        odds_proba, odds_length = self.equalized_odds_proba(data_obj, model)
        self.stat_parity = self.statistical_parity_eval(stat_proba, stat_length)
        self.eq_odds = self.equalized_odds_eval(odds_proba, odds_length)

    def add_specific_x_data(self,idx,x,original_x,x_label,x_target,data_obj):
        """
        Method that calculates and stores x data found in the Evaluator
        """
        self.idx = idx
        self.x = x
        self.x_original_pd = original_x
        self.x_pd = pd.DataFrame(data=[self.x], index=[self.idx], columns=self.data_cols)
        self.normal_x = self.x_pd
        self.x_label = x_label
        self.x_target = x_target
        self.x_accuracy = self.accuracy()[0]
        df_x_row = pd.DataFrame(data = [[self.x_pd, self.normal_x, self.x_original_pd,
                                         self.x_label, self.x_target, self.x_accuracy]], columns = self.x_columns)
        self.all_x_data = self.all_x_data.append(df_x_row)

    def inverse_transform_original(self, jce_instance):
        """
        Method to transform a JCE-ready instance to original features
        """
        instance_index = jce_instance.index
        original_instance_pd = pd.DataFrame(index=instance_index)
        if len(self.oh_bin_enc_cols) > 0:
            instance_bin = self.oh_bin_enc.inverse_transform(jce_instance[self.oh_bin_enc_cols])
            instance_bin_pd = pd.DataFrame(data=instance_bin,index=instance_index,columns=self.binary)
            original_instance_pd = pd.concat((original_instance_pd,instance_bin_pd),axis=1)
        if len(self.numerical) > 0:
            instance_num = self.scaler.inverse_transform(jce_instance[self.numerical])
            instance_num_pd = pd.DataFrame(data=instance_num,index=instance_index,columns=self.numerical)
            original_instance_pd = pd.concat((original_instance_pd,instance_num_pd),axis=1)
        if len(self.oh_cat_enc_cols) > 0:
            instance_cat = self.oh_cat_enc.inverse_transform(jce_instance[self.oh_cat_enc_cols])
            instance_cat_pd = pd.DataFrame(data=instance_cat,index=instance_index,columns=self.categorical)
            original_instance_pd = pd.concat((original_instance_pd,instance_cat_pd),axis=1)
        return original_instance_pd

    def add_specific_cf_data(self,data_obj,cf_method_name,cf,cf_time):
        """
        Method that calculates and stores a cf method result and performance metrics into the Pandas DataFrame found in the Evaluator
        """
        self.cf_method_name = cf_method_name
        self.cf = cf
        if cf is not None and not np.isnan(np.sum(cf)):
            if isinstance(self.cf,pd.DataFrame):
                self.cf_pd = self.cf
            elif isinstance(self.cf,pd.Series):
                cf_np = self.cf.to_numpy()
                self.cf_pd = pd.DataFrame(data = [cf_np], index = [self.idx], columns = data_obj.jce_all_cols) 
            else:
                self.cf_pd = pd.DataFrame(data = [self.cf], index = [self.idx], columns = data_obj.jce_all_cols)
            sorted_train_cf = sort_data_distance(cf,data_obj.jce_train_np,data_obj.train_target)
            for i in sorted_train_cf:
                if i[2] != self.x_label:
                    nn_to_cf = i[0]
                    label_nn_to_cf = i[2]
                    break
            normal_cf = self.x_pd
            self.cf_valid = True
        else:
            self.cf = self.search_desired_class_penalize(data_obj)
            self.cf_pd = pd.DataFrame(data = [self.cf], index = [self.idx], columns = data_obj.jce_all_cols)
            nn_to_cf = self.cf_pd
            label_nn_to_cf = 1 - self.undesired_class
            self.nn_to_cf, self.cf_label  = nn_to_cf, label_nn_to_cf
            normal_cf = self.cf_pd
            self.cf_valid = False
        self.original_cf_pd = self.inverse_transform_original(self.cf_pd)
        self.normal_cf = normal_cf
        self.proximity()
        self.feasibility()
        self.sparsity(data_obj)
        self.cf_time = cf_time
        df_cf_row = pd.DataFrame(data = [[self.cf_method_name, self.cf_pd, self.normal_cf,
                                          self.original_cf_pd, self.cf_proximity, self.cf_feasibility,
                                          self.cf_sparsity, self.cf_valid, self.cf_time]], columns = self.eval_columns)
        self.all_cf_data = self.all_cf_data.append(df_cf_row)

    def accuracy(self):
        """
        Method that estimates accuracy between the target and the predicted label of x
        """
        return self.x_target == self.x_label

    def proximity(self):
        """
        Method that calculates the distance between the instance of interest and the counterfactual given
        """
        if (self.x_pd.columns == self.cf_pd.columns).all():
            self.cf_proximity = np.round_(euclidean(self.x,self.cf_pd.to_numpy()),3)
        else:
            instance_x_copy = copy.deepcopy(self.x_pd)
            instance_x_copy = instance_x_copy[self.cf_pd.columns]
            self.cf_proximity = np.round_(euclidean(instance_x_copy.to_numpy(),self.cf_pd.to_numpy()),3)

    def feasibility(self):
        """
        Method that indicates whether cf is a feasible counterfactual with respect to x and the feature mutability
        """
        toler = 0.000001
        feasibility = True
        for i in range(len(self.feat_type)):
            if self.feat_type[i] == 'bin':
                if not np.isclose(self.cf[i],[0,1],atol=toler).any():
                    feasibility = False
                    break
            elif self.feat_type[i] == 'num-ord':
                possible_val = np.linspace(0,1,int(1/self.feat_step[i]+1),endpoint=True)
                if not np.isclose(self.cf[i],possible_val,atol=toler).any():
                    feasibility = False
                    break
            else:
                if self.cf[i] < 0-toler or self.cf[i] > 1+toler:
                    feasibility = False
                    break
            vector = self.cf - self.x
            if self.feat_dir[i] == 0 and vector[i] != 0:
                feasibility = False
                break
            elif self.feat_dir[i] == 'pos' and vector[i] < 0:
                feasibility = False
                break
            elif self.feat_dir[i] == 'neg' and vector[i] > 0:
                feasibility = False
                break
        if not np.array_equal(self.x[np.where(self.feat_mutable == 0)],self.cf[np.where(self.feat_mutable == 0)]):
            feasibility = False
        self.cf_feasibility = feasibility

    def sparsity(self,data):
        """
        Function that calculates sparsity for a given counterfactual according to x
        Sparsity: 1 - the fraction of features changed in the cf. Takes the value of 1 if the number of changed features is 1.
        """
        unchanged_features = np.sum(np.equal(self.x,self.cf))
        categories_feat_changed = data.feat_cat[np.where(np.equal(self.x,self.cf) == False)[0]]
        len_categories_feat_changed_unique = len([i for i in np.unique(categories_feat_changed) if 'cat' in i])
        unchanged_features += len_categories_feat_changed_unique
        n_changed = len(self.x) - unchanged_features
        if n_changed == 1:
            sparsity = 1.000
        else:
            sparsity = np.round_(1 - n_changed/len(self.x),3)
        self.cf_sparsity = sparsity
   
    def evaluate_cf_models(self, x_jce_np, x_label, data, model, epsilon_ft, carla_model, x_carla_pd):
        """
        Method that evaluates all the specified models
        """
        for model_str in self.models:
            if 'mutable' in model_str:
                mutability_check = False
            else:
                mutability_check = True
            if 'nn' in model_str:
                cf, run_time = nn(x_jce_np,x_label,data,mutability_check)
            elif 'mo' in model_str:
                cf, run_time = min_obs(x_jce_np,x_label,data,mutability_check)
            elif 'rt' in model_str:
                cf, run_time = rf_tweak(x_jce_np,x_label,model.jce_rf,data,True,mutability_check)
            elif 'cchvae' in model_str:
                cf, run_time = cchvae_function(data, carla_model, x_carla_pd)
            
            # WORK IN PROGRESS:
            # elif 'ft' in model_str:
            #     cf, run_time = feat_tweak(x_jce_np,model.jce_rf,epsilon_ft)
            # elif 'face' in model_str:
            #     cf, run_time = face_function(data, carla_model, x_carla_pd)
            # elif 'gs' in model_str:
            #     cf, run_time = gs_function(data, carla_model, x_carla_pd)
            # elif 'dice' in model_str:
            #     cf, run_time = dice_function(data, carla_model, x_carla_pd)
            # elif 'juice' in model_str:
            #     results = JUICE(x_jce_np,x_label,data,model.jce_sel,'proximity',mutability_check)
            #     cf, run_time = results[0], results[4]

            print(f'  {model_str} (time (s): {np.round_(run_time,2)})')
            print(f'---------------------------')
            self.add_specific_cf_data(data,model_str,cf,run_time,model)
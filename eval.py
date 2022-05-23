"""
Evaluation algorithms
"""

"""
Imports
"""

import numpy as np
import pandas as pd
from naj import verify_justification
from support import euclidean, sort_data_distance
from nt import nn
from mo import min_obs
from ft import feat_tweak
from rt import rf_tweak
from juice import JUICE
from carla import DataCatalog, MLModelCatalog
from carla.recourse_methods import Face, Dice, CCHVAE, GrowingSpheres
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
        (5) train:                     Training dataset from the Dataset object,
        (6) train_target:              Training dataset target from the Dataset object,
        (7) idx:                       Index of the testing instanceTraining dataset targets,
        (8) x:                         Instance of interest,
        (9) x_label:                   Predicted label of the instance of interest,
       (10) normal_x:                  Normal instance after transformation (inverse scaling and inverse encoding),
       (11) cf_method_name:            Method used to obtain the cf,
       (12) cf:                        Counterfactual instance,
       (13) normal_cf:                 Normal counterfactual instance after transformation (inverse scaling and inverse encoding),
       (14) nn_to_cf:                  Nearest neighbor to the counterfactual found,
       (15) cf_label:                  Counterfactual label (which is the same of the nn_to_cf)
       (16) cf_proximity:              Proximity as distance between the counterfactual and the instance of interest,
       (17) cf_feasibility:            Feasibility of the counterfactual according to the dataset,
       (18) cf_sparsity:               Sparsity of the counterfactual w.r.t. the instance of interest,
       (19) cf_time:                   Computational time of the counterfactual,
       (20) cf_justification:          Justification of the counterfactual,
       (21) justifier_instance:        Justifier instance to the counterfactual,
       (22) normal_justifier_instance: Justifier instance to the counterfactual after transformation (inverse scaling and inverse encoding),
       (23) found_justifiable_cf:      The JCF algorithm found a justifiable counterfactual according to the model,
       (24) n_feat:                    Number of features to generate per feature in the continuous feature space
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
        # self.eval_columns = ['instance_index','cf_method','cf','normal_cf',
        #                      'proximity','feasibility','sparsity',
        #                      'justification','justifier','normal_justifier','time']
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
        # penalize_instance = np.mean(train_desired_class, axis=0)
        return penalize_instance

    def statistical_parity_eval(self, data_obj, model_obj):
        """
        Method that calculates the Statistical Parity measure for each of the protected feature groups given a model
        """
        stat_parity = {}
        test_data = data_obj.test_pd
        test_data_jce = data_obj.jce_test_pd
        test_data_jce_index = test_data_jce.index
        pred = model_obj.jce_sel.predict(test_data_jce)
        pred_pd = pd.DataFrame(data=pred, index=test_data_jce_index, columns=['prediction'])
        test_data_with_pred = pd.concat((test_data, pred_pd),axis=1)
        for i in self.protected_feat.keys():
            feat_i_values = test_data_with_pred[i].unique()
            val_dict = {}
            for j in feat_i_values:
                val_name = self.protected_feat[i][np.round(j,2)]
                total_feat_i_val_j = len(test_data_with_pred[test_data_with_pred[i] == j])
                total_feat_i_val_j_desired_pred = len(test_data_with_pred[test_data_with_pred[i] == j & test_data_with_pred['prediction'] == self.desired_class])
                stat_parity_feat_val = total_feat_i_val_j_desired_pred / total_feat_i_val_j
                val_dict[val_name] = stat_parity_feat_val
            stat_parity[i] = val_dict
        return stat_parity

    def equalized_odds_eval(self, data_obj, model_obj):
        """
        Method that calculates the Equalized Odds measure for each of the protected feature groups given a model
        """
        eq_odds = {}
        test_data = data_obj.test_pd
        test_data_jce = data_obj.jce_test_pd
        test_data_jce_index = test_data_jce.index
        test_target = data_obj.test_target
        pred = model_obj.jce_sel.predict(test_data_jce)
        pred_pd = pd.DataFrame(data=pred, index=test_data_jce_index, columns=['prediction'])
        target_pd = pd.DataFrame(data=test_target, index=test_data_jce_index, columns=['target'])
        test_data_with_pred_target = pd.concat((test_data, pred_pd, target_pd),axis=1)
        for i in self.protected_feat.keys():
            feat_i_values = test_data_with_pred_target[i].unique()
            val_dict = {}
            for j in feat_i_values:
                val_name = self.protected_feat[i][np.round(j,2)]
                total_feat_i_val_j_ground_0 = len(test_data_with_pred_target[(test_data_with_pred_target[i] == j) & (test_data_with_pred_target['target'] == 0)])
                total_feat_i_val_j_ground_1 = len(test_data_with_pred_target[(test_data_with_pred_target[i] == j) & (test_data_with_pred_target['target'] == 1)])
                total_feat_i_val_j_desired_pred = len(test_data_with_pred_target[test_data_with_pred_target[i] == j & test_data_with_pred_target['prediction'] == self.desired_class])
                stat_parity_feat_val = total_feat_i_val_j_desired_pred / total_feat_i_val_j
                val_dict[val_name] = stat_parity_feat_val
            eq_odds[i] = val_dict
        return eq_odds
    
        #     self.statistical_parity = self.statistical_parity_eval()
        # self.equalized_odds = self.equalized_odds_eval()

    def add_specific_x_data(self,idx,x,original_x,x_label,x_target,data_obj):
        """
        Method that calculates and stores x data found in the Evaluator
        """
        self.idx = idx
        self.x = x
        self.original_x = original_x
        self.x_pd = pd.DataFrame(data=[self.x], index=[self.idx], columns=self.data_cols)
        self.x_original_pd = pd.DataFrame(data=[self.original_x], index=[self.idx], columns=self.raw_data_cols)
        self.normal_x = data_obj.adjust_to_mace_format(self.x_pd)
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

    def add_specific_cf_data(self,data_obj,cf_method_name,cf,cf_time,model,justifier_instance = None,cf_justification = None,found_justifiable_jcf = None):
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
            normal_cf = data_obj.adjust_to_mace_format(self.cf_pd)
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
        # if cf_justification is None:
        #     self.justification(model,data_obj)
        # else:
        #     self.cf_justification = cf_justification
        #     self.justifier_instance = justifier_instance
        #     justifier_instance_pd = pd.DataFrame(data = [self.justifier_instance], index = [self.idx], columns = data_obj.jce_all_cols) 
        #     self.normal_justifier_instance = data_obj.adjust_to_mace_format(justifier_instance_pd)
        # df_cf_row = pd.DataFrame(data = [[self.cf_method_name, self.cf_pd, self.normal_cf,
        #                                   self.cf_proximity, self.cf_feasibility, self.cf_sparsity,
        #                                   self.cf_justification, self.justifier_instance, self.normal_justifier_instance, self.cf_time]],
        #                                   columns = self.eval_columns)
        # self.found_justifiable_cf = found_justifiable_jcf
        # df_cf_row = pd.DataFrame(data = [[self.idx, self.cf_method_name, self.cf_pd, self.normal_cf,
        #                                   self.cf_proximity, self.cf_feasibility, self.cf_sparsity,
        #                                   self.cf_justification, self.justifier_instance, self.normal_justifier_instance, self.cf_time]],
        #                                   columns = self.eval_columns)
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
        self.cf_proximity = np.round_(euclidean(self.x,self.cf_pd.to_numpy()),3)

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

    def justification(self,model,data):
        """
        Method that finds whether the instance of interest is justified or not
        """
        self.justifier_instance, self.cf_justification = verify_justification(self,model,data)
        self.justifier_instance_pd = pd.DataFrame(data=[self.justifier_instance],columns=data.jce_all_cols)
        self.normal_justifier_instance = data.adjust_to_mace_format(self.justifier_instance_pd)
    
    def evaluate_cf_models(self, x_jce_np, x_label, data, model, epsilon_ft):
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
            elif 'ft' in model_str:
                cf, run_time = feat_tweak(x_jce_np,model.jce_rf,epsilon_ft)
            elif 'rt' in model_str:
                cf, run_time = rf_tweak(x_jce_np,x_label,model.jce_rf,data,True,mutability_check)
            print(f'  {model_str} (time (s): {np.round_(run_time,2)})')
            print(f'---------------------------')
            self.add_specific_cf_data(data,model_str,cf,run_time,model)
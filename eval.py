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
        self.oh_bin_enc, self.oh_bin_enc_cols  = data_obj.bin_enc, data_obj.bin_enc_cols
        self.oh_cat_enc, self.oh_cat_enc_cols  = data_obj.cat_enc, data_obj.cat_enc_cols
        self.scaler = data_obj.scaler
        self.data_cols = data_obj.transformed_cols
        self.raw_data_cols = data_obj.train_df.columns
        self.undesired_class = data_obj.undesired_class
        self.desired_class = 1 - self.undesired_class
        self.n_feat = n_feat
        self.x_columns = ['x','normal_x','original_x',
                          'x_pred','x_target','accuracy']
        self.eval_columns = ['cf','normal_cf','original_cf',
                             'proximity','feasibility','sparsity','valid','time']
        self.all_x_data = pd.DataFrame(columns=self.x_columns)
        self.all_cf_data = pd.DataFrame(columns=self.eval_columns)
    
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
        pred = model_obj.jce_sel.predict(test_data_transformed)
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

    def add_specific_x_data(self, idx, x, original_x, x_label, x_target):
        """
        DESCRIPTION:        Calculates and stores x data found in the Evaluator

        INPUT:
        idx:                Index of the instance x
        x:                  Instance of interest
        original_x:         Instance of interest in original format (before normalization and encoding)
        x_label:            Predicted label of the instance of interest
        x_target:           Ground truth label of the instance of interest

        OUTPUT: (None: stored as class attributes)
        """
        self.idx = idx
        self.x = x
        self.x_original_df = original_x
        self.x_df = pd.DataFrame(data=[self.x], index=[self.idx], columns=self.data_cols)
        self.normal_x = self.x_df
        self.x_label = x_label
        self.x_target = x_target
        self.x_accuracy = self.accuracy()[0]
        df_x_row = pd.DataFrame(data = [[self.x_df, self.normal_x, self.x_original_df,
                                         self.x_label, self.x_target, self.x_accuracy]], columns = self.x_columns)
        self.all_x_data = self.all_x_data.append(df_x_row)

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
        if len(self.oh_bin_enc_cols) > 0:
            instance_bin = self.oh_bin_enc.inverse_transform(instance[self.oh_bin_enc_cols])
            instance_bin_pd = pd.DataFrame(data=instance_bin,index=instance_index,columns=self.binary)
            original_instance_df = pd.concat((original_instance_df,instance_bin_pd),axis=1)
        if len(self.numerical) > 0:
            instance_num = self.scaler.inverse_transform(instance[self.numerical])
            instance_num_pd = pd.DataFrame(data=instance_num,index=instance_index,columns=self.numerical)
            original_instance_df = pd.concat((original_instance_df,instance_num_pd),axis=1)
        if len(self.oh_cat_enc_cols) > 0:
            instance_cat = self.oh_cat_enc.inverse_transform(instance[self.oh_cat_enc_cols])
            instance_cat_pd = pd.DataFrame(data=instance_cat,index=instance_index,columns=self.categorical)
            original_instance_df = pd.concat((original_instance_df,instance_cat_pd),axis=1)
        return original_instance_df

    def add_specific_cf_data(self, data_obj, cf, cf_time):
        """
        DESCRIPTION:        Calculates and stores a cf method result and performance metrics into the Pandas DataFrame found in the Evaluator

        INPUT:
        data_obj:           Dataset object
        cf:                 Counterfactual instance obtained
        cf_time:            Run time for the counterfactual method used
        """
        self.cf = cf
        if cf is not None and not np.isnan(np.sum(cf)):
            if isinstance(self.cf, pd.DataFrame):
                self.cf_df = self.cf
            elif isinstance(self.cf, pd.Series):
                cf_np = self.cf.to_numpy()
                self.cf_df = pd.DataFrame(data=[cf_np], index=[self.idx], columns=data_obj.transformed_cols) 
            else:
                self.cf_df = pd.DataFrame(data = [self.cf], index = [self.idx], columns = data_obj.transformed_cols)
            sorted_train_cf = sort_data_distance(cf, data_obj.transformed_train_np, data_obj.train_target)
            for i in sorted_train_cf:
                if i[2] != self.x_label:
                    nn_to_cf = i[0]
                    label_nn_to_cf = i[2]
                    break
            normal_cf = self.x_df
            self.cf_valid = True
        else:
            self.cf = self.search_desired_class_penalize(data_obj)
            self.cf_df = pd.DataFrame(data=[self.cf], index=[self.idx], columns=data_obj.transformed_cols)
            nn_to_cf = self.cf_df
            label_nn_to_cf = 1 - self.undesired_class
            self.nn_to_cf, self.cf_label = nn_to_cf, label_nn_to_cf
            normal_cf = self.cf_df
            self.cf_valid = False
        self.original_cf_df = self.inverse_transform_original(self.cf_df)
        self.normal_cf = normal_cf
        self.proximity()
        self.feasibility()
        self.sparsity(data_obj)
        self.cf_time = cf_time
        df_cf_row = pd.DataFrame(data = [[self.cf_df, self.normal_cf, self.original_cf_df, 
                                          self.cf_proximity, self.cf_feasibility, self.cf_sparsity, self.cf_valid, self.cf_time]], columns = self.eval_columns)
        self.all_cf_data = self.all_cf_data.append(df_cf_row)

    def accuracy(self):
        """
        DESCRIPTION:        Estimates accuracy between the target and the predicted label of x

        INPUT:
        self

        OUTPUT:
        acc:                Accuracy (agreement) between the predicted and ground truth labels 
        """
        acc = self.x_target == self.x_label
        return acc

    def proximity(self):
        """
        DESCRIPTION:        Calculates the distance between the instance of interest and the counterfactual given

        INPUT:
        self

        OUTPUT: (None: stored as class attributes)
        """
        if (self.x_pd.columns == self.cf_pd.columns).all():
            self.cf_proximity = np.round_(euclidean(self.x,self.cf_pd.to_numpy()),3)
        else:
            instance_x_copy = copy.deepcopy(self.x_pd)
            instance_x_copy = instance_x_copy[self.cf_pd.columns]
            self.cf_proximity = np.round_(euclidean(instance_x_copy.to_numpy(),self.cf_pd.to_numpy()),3)

    def feasibility(self):
        """
        DESCRIPTION:        Indicates whether cf is a feasible counterfactual with respect to x and the feature mutability

        INPUT:
        self

        OUTPUT: (None: stored as class attributes)
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

    def sparsity(self, data_obj):
        """
        DESCRIPTION:        Calculates sparsity for a given counterfactual according to x. Sparsity is 1 - the fraction of features changed in the cf. Takes the value of 1 if the number of changed features is 1

        INPUT:
        data_obj:           Dataset object

        OUTPUT: (None: stored as class attributes)
        """
        unchanged_features = np.sum(np.equal(self.x,self.cf))
        categories_feat_changed = data_obj.feat_cat[np.where(np.equal(self.x,self.cf) == False)[0]]
        len_categories_feat_changed_unique = len([i for i in np.unique(categories_feat_changed) if 'cat' in i])
        unchanged_features += len_categories_feat_changed_unique
        n_changed = len(self.x) - unchanged_features
        if n_changed == 1:
            sparsity = 1.000
        else:
            sparsity = np.round_(1 - n_changed/len(self.x),3)
        self.cf_sparsity = sparsity
   
    def evaluate_cf_models(self, x_np, x_label, data_obj, model_obj, epsilon_ft, carla_model, x_carla_df):
        """
        DESCRIPTION:        Evaluates the specific counterfactual method on the isntance of interest

        INPUT:
        x_np:               Instance of interest in Numpy array format
        x_label:            Predicted label of the instance of interest
        data_obj:           Dataset object
        model_obj:          Model object
        epsilon_ft:         Parameter for the Feature Tweaking counterfactual method
        carla_model:        CARLA framework classifier model
        x_carla_df:         Instance of interest in the CARLA framework format

        OUTPUT: (None: stored as class attributes)
        """
        if 'mutable' in self.method_name:
            mutability_check = False
        else:
            mutability_check = True
        if 'nn' in self.method_name:
            cf, run_time = nn(x_np ,x_label, data_obj, mutability_check)
        elif 'mo' in self.method_name:
            cf, run_time = min_obs(x_np, x_label, data_obj, mutability_check)
        elif 'rt' in self.method_name:
            cf, run_time = rf_tweak(x_np, x_label, model_obj.rf, data_obj, True, mutability_check)
        elif 'cchvae' in self.method_name:
            cf, run_time = cchvae_function(data_obj, carla_model, x_carla_df)
            
        # WORK IN PROGRESS:
        # elif 'ft' in self.method_name:
        #     cf, run_time = feat_tweak(x_np, model_obj.rf, epsilon_ft)
        # elif 'face' in self.method_name:
        #     cf, run_time = face_function(data_obj, carla_model, x_carla_df)
        # elif 'gs' in self.method_name:
        #     cf, run_time = gs_function(data_obj, carla_model, x_carla_df)
        # elif 'dice' in self.method_name:
        #     cf, run_time = dice_function(data_obj, carla_model, x_carla_df)
        # elif 'juice' in self.method_name:
        #     results = JUICE(x_np, x_label, data_obj, model_obj.sel, 'proximity', mutability_check)
        #     cf, run_time = results[0], results[4]

        print(f'  {self.method_name} (time (s): {np.round_(run_time,2)})')
        print(f'---------------------------')
        self.add_specific_cf_data(data_obj, cf, run_time, model_obj)
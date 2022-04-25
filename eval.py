"""
Evaluation algorithms
"""

"""
Imports
"""

import numpy as np
import pandas as pd
from naj import verify_justification
from address_distance_feasibility import euclidean, sort_data_distance

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

    def __init__(self,data_obj,n_feat):
        self.data_name = data_obj.name
        self.feat_type = data_obj.feat_type
        self.feat_mutable = data_obj.feat_mutable
        self.feat_cost = data_obj.feat_cost
        self.feat_step = data_obj.feat_step
        self.n_feat = n_feat
        self.data_cols = data_obj.jce_all_cols
        self.eval_columns = ['index','x','normal_x','x_label',
                             'cf_method','cf','normal_cf','proximity',
                             'feasibility','sparsity','justification','justifier','normal_justifier','time']
        self.all_cf_data = pd.DataFrame(columns=self.eval_columns)
    
    def add_specific_x_data(self,idx,x,x_label,data_obj):
        """
        Method that calculates and stores x data found in the Evaluator
        """
        self.idx = idx
        self.x = x
        self.x_pd = pd.DataFrame(data = [self.x], index = [idx], columns = data_obj.jce_all_cols)
        self.normal_x = data_obj.adjust_to_mace_format(self.x_pd)
        self.x_label = x_label

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
            self.nn_to_cf, self.cf_label  = nn_to_cf, label_nn_to_cf
            self.normal_cf = data_obj.adjust_to_mace_format(self.cf_pd)
            self.proximity()
            self.feasibility()
            self.sparsity(data_obj)
            self.cf_time = cf_time
            if cf_justification is None:
                self.justification(model,data_obj)
            else:
                self.cf_justification = cf_justification
                self.justifier_instance = justifier_instance
                justifier_instance_pd = pd.DataFrame(data = [self.justifier_instance], index = [self.idx], columns = data_obj.jce_all_cols) 
                self.normal_justifier_instance = data_obj.adjust_to_mace_format(justifier_instance_pd)
        else:
            self.cf_pd = None
            self.nn_to_cf = None
            self.label_nn_to_cf = None
            self.normal_cf = np.nan
            self.cf_proximity = np.nan
            self.cf_feasibility = np.nan
            self.cf_sparsity = np.nan
            self.cf_time = np.nan
            self.cf_justification = np.nan
            self.justifier_instance = np.nan
            self.justifier_instance_pd = np.nan
            self.normal_justifier_instance = np.nan
        self.found_justifiable_cf = found_justifiable_jcf
        df_cf_row = pd.DataFrame(data = [[self.idx,self.x,self.normal_x,self.x_label,
                                 self.cf_method_name,self.cf,self.normal_cf,self.cf_proximity,
                                 self.cf_feasibility,self.cf_sparsity,self.cf_justification,self.justifier_instance,self.normal_justifier_instance,self.cf_time]],
                                 columns = self.eval_columns)
        self.all_cf_data = self.all_cf_data.append(df_cf_row)

    def proximity(self):
        """
        Method that calculates the distance between the instance of interest and the counterfactual given
        """
        self.cf_proximity = np.round_(euclidean(self.normal_x.to_numpy(),self.normal_cf.to_numpy()),3)

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
        # print(f'after for: {feasibility}, {not np.array_equal(x[np.where(mutable_feat == 0)],cf[np.where(mutable_feat == 0)])}')
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
import numpy as np
import pandas as pd
import pickle
from support import path_here, results_cf_obj_dir

def load_obj(file_name):
    """
    Method to read an Evaluator object containing the evaluation results for all the instances of a given dataset
    """
    with open(results_cf_obj_dir+file_name, 'rb') as input:
        evaluator_obj = pickle.load(input)
    return evaluator_obj

def get_df(eval_obj,method):
    """
    Method that obtains the DataFrame containing all data from the data_str Dataset and the method specified
    Input eval_obj: Evaluator object
    Input method: Method of interest
    Output df: DataFrame containing the instances and corresponding CFs from the method specified
    """
    eval_df = eval_obj.all_cf_data
    eval_df_method = eval_df[eval_df['cf_method'] == method]
    return eval_df_method

datasets = ['compass','credit','adult','german','heart']  # Name of the dataset to be analyzed ['synthetic_severe_disease','synthetic_athlete','ionosphere','compass','credit','adult','german','heart']
methods_to_run = ['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice'] #['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice']


self.feat_protected
self.eval_columns = ['index','x','normal_x','x_label',
                             'cf_method','cf','normal_cf','proximity',
                             'feasibility','sparsity','justification','justifier','normal_justifier','time']
self.all_cf_data = pd.DataFrame(columns=self.eval_columns)

for data_str in datasets:
    eval_obj = load_obj(data_str)
    protected_feat = eval_obj.feat_protected
    for metric in ['proximity','sparsity','accuracy']:
        for prot_feat in list(protected_feat.keys()):
            for method in methods_to_run:
                df_method = get_df(eval_obj,method)




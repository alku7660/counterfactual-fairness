"""
Imports
"""

import ast
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import copy

def best_model_params(grid_search_pd, data_str):
    """
    DESCRIPTION:        Delivers the best model and its parameters according to the Grid Search done
    
    INPUT:
    grid_search_pd:     DataFrame containing the parameters of the models tested in the Grid Search
    data_str:           String containing the name of the dataset
    
    OUTPUT:
    best:               The name of the best performing model
    params_best:        The parameters of the best performing model
    params_rf:          The parameters of the RF model
    """
    if data_str in ['adult','kdd_census','dutch','bank','student']:
        best = 'rf'
    elif data_str in ['credit','german','diabetes','oulad','law','compass']:
        best = 'mlp'
    params_best = ast.literal_eval(grid_search_pd.loc[(data_str,best), 'params'])[0]
    params_rf = ast.literal_eval(grid_search_pd.loc[(data_str,'rf'), 'params'])[0]
    return best, params_best, params_rf

def clf_model(model_str, best_params, rf_params, train_data, train_target):
    """
    DESCRIPTION:        Outputs the best trained model according to Grid Search done
    
    INPUT:
    model_str:          The name of the best performing model
    best_params:        Parameters of the best performing model
    rf_params:          Parameters of the RF model
    train_data:         Training dataset
    train_target:       Target of the training dataset

    OUTPUT:
    model:              Trained best performing model
    """
    random_st = 54321 
    if model_str == 'svm':
        best_C = best_params['C']
        best_coef0 = best_params['coef0']
        best_degree = best_params['degree']
        best_kernel = best_params['kernel']
        best_model = svm.SVC(C=best_C, coef0=best_coef0, degree=best_degree, kernel=best_kernel)
        best_model.fit(train_data,train_target)
        rf_model = RandomForestClassifier(max_depth=rf_params['max_depth'], min_samples_leaf=rf_params['min_samples_leaf'], min_samples_split=rf_params['min_samples_split'], n_estimators=rf_params['n_estimators']) 
        rf_model.fit(train_data,train_target)
    elif model_str == 'dt':
        best_max_depth = best_params['max_depth']
        best_min_samples_leaf = best_params['min_samples_leaf']
        best_min_samples_split = best_params['min_samples_split']
        best_model = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf, min_samples_split=best_min_samples_split)
        best_model.fit(train_data,train_target)
        rf_model = RandomForestClassifier(max_depth=rf_params['max_depth'], min_samples_leaf=rf_params['min_samples_leaf'], min_samples_split=rf_params['min_samples_split'], n_estimators=rf_params['n_estimators']) 
        rf_model.fit(train_data,train_target)
    elif model_str == 'mlp':
        best_activation = best_params['activation']
        best_hidden_layer_sizes = best_params['hidden_layer_sizes']
        best_solver = best_params['solver']
        best_model = MLPClassifier(activation=best_activation, hidden_layer_sizes=best_hidden_layer_sizes, solver=best_solver, random_state=random_st)
        best_model.fit(train_data,train_target)
        rf_model = RandomForestClassifier(max_depth=rf_params['max_depth'], min_samples_leaf=rf_params['min_samples_leaf'], min_samples_split=rf_params['min_samples_split'], n_estimators=rf_params['n_estimators']) 
        rf_model.fit(train_data,train_target)
    elif model_str == 'rf':
        best_max_depth = best_params['max_depth']
        best_min_samples_leaf = best_params['min_samples_leaf']
        best_min_samples_split = best_params['min_samples_split']
        best_n_estimators = best_params['n_estimators']
        best_model = RandomForestClassifier(max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf, min_samples_split=best_min_samples_split, n_estimators=best_n_estimators)
        best_model.fit(train_data,train_target)
        rf_model = copy.deepcopy(best_model)  
    return best_model, rf_model
"""
Imports
"""
import pandas as pd
import ast
from support import results_grid_search
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

class Model:

    def __init__(self, data) -> None:
        self.model, self.rf_model = self.train_model(data)
    
    def best_model_params(self, grid_search_pd, data_str):
        """
        Method that delivers the best model and its parameters according to the Grid Search done
        """
        if data_str in ['bank','ionosphere','german','dutch','student','heart','kdd_census']:
            best = 'rf'
        elif data_str in ['adult','compass','credit','diabetes','german','law','oulad','synthetic_athlete','synthetic_disease']:
            best = 'mlp'
        params_best = ast.literal_eval(grid_search_pd.loc[(data_str, best), 'params'])[0]
        params_rf = ast.literal_eval(grid_search_pd.loc[(data_str, 'rf'), 'params'])[0]
        return best, params_best, params_rf

    def classifier(self, model_str, best_params, params_rf, train_data, train_target, test_data, test_target):
        """
        Method that outputs the best trained model according to Grid Search done
        """
        random_st = 54321
        rf_max_depth = params_rf['max_depth']
        rf_min_samples_leaf = params_rf['min_samples_leaf']
        rf_min_samples_split = params_rf['min_samples_split']
        rf_n_estimators = params_rf['n_estimators']
        rf_model = RandomForestClassifier(max_depth=rf_max_depth, min_samples_leaf=rf_min_samples_leaf, min_samples_split=rf_min_samples_split, n_estimators=rf_n_estimators)
        rf_model.fit(train_data,train_target)
        if model_str == 'mlp':
            best_activation = best_params['activation']
            best_hidden_layer_sizes = best_params['hidden_layer_sizes']
            best_solver = best_params['solver']
            best_model = MLPClassifier(activation=best_activation, hidden_layer_sizes=best_hidden_layer_sizes, solver=best_solver, random_state=random_st)
            best_model.fit(train_data,train_target)
        elif model_str == 'rf':
            best_model = rf_model
        print(f'Model test F1 score: {f1_score(test_target, best_model.predict(test_data))}')
        return best_model, rf_model

    def train_model(self, data):
        """
        Constructs a model for the dataset using sklearn modules
        """
        grid_search_results = pd.read_csv(results_grid_search+'grid_search.csv', index_col = ['dataset','model'])
        sel_model_str, params_best, params_rf = self.best_model_params(grid_search_results, data.name)
        best_model, rf_model = self.classifier(sel_model_str, params_best, params_rf, data.transformed_train_np, data.train_target, data.transformed_test_np, data.test_target)
        return best_model, rf_model
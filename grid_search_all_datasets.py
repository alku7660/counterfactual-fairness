"""
Imports
"""
import warnings
warnings.filterwarnings("ignore")
import itertools
from sklearn import svm
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from data_load import load_model_dataset
import pandas as pd
import numpy as np

path_here = os.path.abspath('')
datasets = ['kdd_census','dutch','bank'] # Name of the dataset to be analyzed ['synthetic_simple','synthetic_severe_disease','synthetic_athlete','compass','credit','adult','german','heart'] 
seed_int = 54321             # Seed integer value
train_fraction = 0.95
step = 0.01

np.random.seed(seed_int)

index_result = pd.MultiIndex.from_tuples(list(itertools.product(datasets, ['svm','dt','mlp','rf'])), names=['dataset','model']) 
results = pd.DataFrame(index=index_result,columns=['params','F1'])
path_none = None

for data_str in datasets:

    data_obj = load_model_dataset(data_str,train_fraction,seed_int,step,path_none)
    if data_str in ['kdd_census','dutch','bank']:
        perc_data = 0.5
        train, train_target = data_obj.jce_train_pd.iloc[0:int(len(data_obj.jce_train_pd)*perc_data)], data_obj.train_target.iloc[0:int(len(data_obj.jce_train_pd)*perc_data)]
    else:
        train, train_target = data_obj.jce_train_pd, data_obj.train_target
    print(f'{data_str} size: {train.shape}')

    param_grid_svm = {'C':[0.01,0.1,1,10], 'kernel':['linear','poly','rbf'], 'degree':[2,3,4,5], 'coef0':[0,0.1,1]}
    clf_search_svm = GridSearchCV(svm.SVC(),param_grid=param_grid_svm,scoring='f1',cv=5,verbose=2.5)
    clf_search_svm.fit(train,train_target)
    results.loc[(data_str,'svm'),'params'] = [clf_search_svm.best_params_]
    results.loc[(data_str,'svm'),'F1'] = clf_search_svm.best_score_

    param_grid_dt = {'max_depth':[2,5,10], 'min_samples_split':[2,5,10], 'min_samples_leaf':[1,3,5]}
    clf_search_dt = GridSearchCV(DecisionTreeClassifier(),param_grid=param_grid_dt,scoring='f1',cv=5,verbose=2.5)
    clf_search_dt.fit(train,train_target)
    results.loc[(data_str,'dt'),'params'] = [clf_search_dt.best_params_]
    results.loc[(data_str,'dt'),'F1'] = clf_search_dt.best_score_

    param_grid_mlp = {'hidden_layer_sizes':[(10,1),(20,1),(50,1),(100,1),
                                            (10,2),(20,2),(50,2),(100,2),
                                            (10,5),(20,5),(50,5),(100,5),
                                            (10,10),(20,10),(50,10),(100,10)],
                                            'activation':['logistic','tanh','relu'],'solver':['lbfgs','sgd','adam']}
    clf_search_mlp = GridSearchCV(MLPClassifier(),param_grid=param_grid_mlp,scoring='f1',cv=5,verbose=2.5)
    clf_search_mlp.fit(train,train_target)
    results.loc[(data_str,'mlp'),'params'] = [clf_search_mlp.best_params_]
    results.loc[(data_str,'mlp'),'F1'] = clf_search_mlp.best_score_

    param_grid_rf = {'n_estimators':[10,20,50,100,200],'max_depth':[2,5,10], 'min_samples_split':[2,5,10], 'min_samples_leaf':[1,3,5]}
    clf_search_rf = GridSearchCV(RandomForestClassifier(),param_grid=param_grid_rf,scoring='f1',cv=5,verbose=2.5)
    clf_search_rf.fit(train,train_target)
    results.loc[(data_str,'rf'),'params'] = [clf_search_rf.best_params_]
    results.loc[(data_str,'rf'),'F1'] = clf_search_rf.best_score_

results.to_csv(str(path_here)+'/Results/grid_search/grid_search_4.csv')

    
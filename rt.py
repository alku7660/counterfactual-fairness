"""
RF Tweaking (RT)
"""

"""
Imports
"""
import time
import numpy as np
from collections import Counter
from nt import nn

# As observed in https://github.com/tony-lind/Example-based-tweaking
"""
Created on Fri Feb 15 20:08:02 2019
@author: Tony Lindgren, tony@dsv.su.se
"""
#import pandas as pd
####
# Method that returns an three arrays, one, containing the number of times the an
# example co-occurs in leaf with ex, two the array of example and three the classes of the examples. 
# For which the following conditions holds true:
# examples share leaf node with example in count number of leafs 
# and is of class whish_class. 
# If classified_as_wish is False (default) is sufficent that the example 
# is of wish_class otherwise it must also be classified as such
def random_forest_tweaking(clf, ex, wish_class, X_train, y_train, classified_as_wish=False):
    # calculate leaf_id_matrix
    leaf_id_mat = clf.apply(X_train)
    # shape of mat
    (rows, cols) = leaf_id_mat.shape 
    # calulate leaf_example_array
    leaf_ex_arr = clf.apply(ex)   
    # for each tree for each training example check if they share leaf node
    cnt = Counter() 
    for col in range(cols - 1):      # trees
        for row in range(rows - 1):  # example id:s
            if leaf_id_mat[row, col] == leaf_ex_arr[0, col]:
                #print("found match at: ", row) 
                cnt[row] += 1
    sorted_cnt = cnt.most_common()
    # filter result on wish_class
    sub_X = np.empty((0, np.size(X_train, 1)))  # define X matrix 
    sub_y = np.empty((0, 1))                    # define y array
    sub_cnt = np.empty((0,1))                   # define cnt array
    for ex, freq in sorted_cnt:
        #print("(Ex, freq): ", ex, freq)
        correct_class = (y_train[ex] == wish_class)
        #print("correct_class: ", correct_class)
        #print("classified_as_wish: ", classified_as_wish)
        if correct_class and classified_as_wish:
            t_pred_c = clf.predict([X_train[ex]])
            #print("t_pred_c: ", t_pred_c)
            if t_pred_c[0] == wish_class:
                #print("Found example of actual wished class which is classified as such with freq: ", freq)
                sub_X = np.vstack([sub_X, X_train[ex]]) 
                sub_y = np.vstack([sub_y, y_train[ex]])  
                sub_cnt = np.vstack([sub_cnt, freq]) 
        elif correct_class:
            #print("Found example of actual wished class with freq:", freq)
            sub_X = np.vstack([sub_X, X_train[ex]]) 
            sub_y = np.vstack([sub_y, y_train[ex]])
            sub_cnt = np.vstack([sub_cnt, freq])        
    return sub_cnt, sub_X, sub_y

def verify_feasibility(x,cf,mutable_feat,feat_type,feat_step,feat_dir,mutability_check):
    """
    Method that indicates whether cf is a feasible counterfactual with respect to x and the feature mutability
    Input x: Instance of interest
    Input cf: Counterfactual to be evaluated
    Input mutable_feat: Vector indicating mutability of the features of x
    Input feat_type: Type of the features used
    Input feat_step: Feature plausible change step size
    Input feat_dir: Directionality of the features
    Input mutability_check: Whether to check or not the mutable features    
    Output: Boolean value indicating whether cf is a feasible counterfactual with regards to x and the feature mutability vector
    """
    toler = 0.000001
    feasibility = True
    vector = cf - x
    for i in range(len(feat_type)):
        if feat_type[i] == 'bin':
            if not np.isclose(cf[i], [0,1],atol=toler).any():
                feasibility = False
                break
        elif feat_type[i] == 'num-ord':
            possible_val = np.linspace(0,1,int(1/feat_step[i]+1),endpoint=True)
            if not np.isclose(cf[i],possible_val,atol=toler).any():
                feasibility = False
                break  
        else:
            if cf[i] < 0-toler or cf[i] > 1+toler:
                feasibility = False
                break
        if feat_dir[i] == 0 and vector[i] != 0:
            feasibility = False
            break
        elif feat_dir[i] == 'pos' and vector[i] < 0:
            feasibility = False
            break
        elif feat_dir[i] == 'neg' and vector[i] > 0:
            feasibility = False
            break
    if mutability_check:
        if not np.array_equal(x[np.where(mutable_feat == 0)],cf[np.where(mutable_feat == 0)]):
            feasibility = False
    return feasibility

# Random Forest Tweaking method (based on Lindgren et al. 2019, found in: https://github.com/tony-lind/Example-based-tweaking)
# Added by Anonymous Author
def rf_tweak(x,x_label,rf_model,data,feasibility_check=True,mutability_check=True):
    """
    Function that returns the Random Forest tweaking counterfactual with respect to instance of interest x
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input rf_model: Random Forest model trained on the training dataset
    Input data: Dataset object
    Output rt_cf: Random Forest Tweaking counterfactual to the instance of interest x
    Output rt_cf_dist_x: Distance from the rt_cf to the instance of interest x
    """
    start_time = time.time()
    x_pred = rf_model.predict(x.reshape(1,-1))
    if x_pred == 0:
        aim = 1
    else:
        aim = 0
    rt_cf_freq, rt_cf_array, rt_cf_label_array = random_forest_tweaking(rf_model,x.reshape(1,-1),aim,data.jce_train_np,data.train_target) 
    found = False
    if len(rt_cf_array) > 0:
        if not feasibility_check:
            rt_cf = rt_cf_array[0]
        else:
            cf_idx = 0
            while not found and cf_idx < len(rt_cf_array):
                feasible = verify_feasibility(x,rt_cf_array[cf_idx],data.feat_mutable,data.feat_type,data.feat_step,data.feat_dir,mutability_check)
                if feasible:
                    rt_cf = rt_cf_array[cf_idx]
                    found = True
                cf_idx += 1
    if len(rt_cf_array) == 0 or not found:
        print(f'No Random Forest Tweaking solution found: Calculating NT solution')
        rt_cf = nn(x,x_label,data,mutability_check)[0]
    end_time = time.time()
    rt_time = end_time - start_time
    return rt_cf, rt_time

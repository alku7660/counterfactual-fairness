"""
Nearest Neighbor (NN)
"""

"""
Imports
"""

import numpy as np
import time

def verify_feasibility(x,cf,mutable_feat,feat_type,feat_step,feat_dir):
    """
    Method that indicates whether cf is a feasible counterfactual with respect to x and the feature mutability
    Input x: Instance of interest
    Input cf: Counterfactual to be evaluated
    Input mutable_feat: Vector indicating mutability of the features of x
    Input feat_type: Type of the features used
    Input feat_step: Feature plausible change step size
    Input feat_dir: Directionality of the features    
    Output: Boolean value indicating whether cf is a feasible counterfactual with regards to x and the feature mutability vector
    """
    toler = 0.000001
    feasibility = True
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
        vector = cf - x
        if feat_dir[i] == 0 and vector[i] != 0:
            feasibility = False
            break
        elif feat_dir[i] == 'pos' and vector[i] < 0:
            feasibility = False
            break
        elif feat_dir[i] == 'neg' and vector[i] > 0:
            feasibility = False
            break
    if not np.array_equal(x[np.where(mutable_feat == 0)],cf[np.where(mutable_feat == 0)]):
        feasibility = False
    return feasibility

def nn(x,x_label,data):
    """
    Function that returns the nearest counterfactual with respect to instance of interest x
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input data: Dataset object
    Output nt_cf: Minimum observable counterfactual to the instance of interest x
    """
    start_time = time.time()
    nt_cf = None
    for i in data.train_sorted:
        if i[2] != x_label and verify_feasibility(x,i[0],data.feat_mutable,data.feat_type,data.feat_step,data.feat_dir) and not np.array_equal(x,i[0]):
            nt_cf = i[0]
            break
    if nt_cf is None:
        print(f'NT could not find a feasible CF!: There is no feasible NN CF available (None output)')
        end_time = time.time()
        return nt_cf, end_time - start_time
    end_time = time.time()
    nn_time = end_time - start_time
    return nt_cf, nn_time

def nn_model(prev_nn,x,x_label,data,model):
    """
    Function that returns the nearest counterfactual with respect to instance of interest x
    Input prev_nn: NT instance using training dataset information and label
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input data: Dataset model
    Input model: Trained model to verify different predicted label from the test dataset
    Output nt_cf: Minimum observable counterfactual to the instance of interest x
    """
    nt_cf = prev_nn
    for i in data.train_sorted:
        if i[2] != x_label and model.predict(i[0].reshape(1,-1)) != x_label and verify_feasibility(x,i[0],data.feat_mutable,data.feat_type,data.feat_step,data.feat_dir) and not np.array_equal(x,i[0]):
            nt_cf = i[0]
            break
    return nt_cf
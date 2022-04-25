"""
Feature Tweaking (FT)
"""

"""
Imports
"""
import time
import numpy as np
import copy

# Adapted from https://github.com/upura/featureTweakPy

def euclidean(x1,x2):
    """
    Calculation of the euclidean distance between two different instances
    Input x1: Instance 1
    Input x2: Instance 2
    Output euclidean distance between x1 and x2
    """
    return np.sqrt(np.sum((x1-x2)**2))

def search_path(estimator, class_labels, aim_label):
    """
    return path index list containing [{leaf node id, inequality symbol, threshold, feature index}].
    estimator: decision tree
    maxj: the number of selected leaf nodes
    """
    """ select leaf nodes whose outcome is aim_label """
    children_left = estimator.tree_.children_left  # information of left child node
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    # leaf nodes ID
    leaf_nodes = np.where(children_left == -1)[0]
    # outcomes of leaf nodes
    leaf_values = estimator.tree_.value[leaf_nodes].reshape(len(leaf_nodes), len(class_labels))
    # select the leaf nodes whose outcome is aim_label
    # leaf_nodes = np.where(leaf_values[:, aim_label] != 0)[0] #(original)
    leaf_nodes = leaf_nodes[np.where(leaf_values[:, aim_label] != 0)[0]] #(added by anonymous author)
    # max_vote_class = np.argmax(leaf_values,axis=1)
    # leaf_nodes = leaf_nodes[np.where(max_vote_class == aim_label)]
    """ search the path to the selected leaf node """
    paths = {}
    for leaf_node in leaf_nodes:
        """ correspond leaf node to left and right parents """
        child_node = leaf_node
        parent_node = -100  # initialize
        parents_left = [] 
        parents_right = [] 
        while (parent_node != 0):
            if (np.where(children_left == child_node)[0].shape == (0, )):
                parent_left = -1
                parent_right = np.where(
                    children_right == child_node)[0][0]
                parent_node = parent_right
            elif (np.where(children_right == child_node)[0].shape == (0, )):
                parent_right = -1
                parent_left = np.where(children_left == child_node)[0][0]
                parent_node = parent_left
            parents_left.append(parent_left)
            parents_right.append(parent_right)
            """ for next step """
            child_node = parent_node
        # nodes dictionary containing left parents and right parents
        paths[leaf_node] = (parents_left, parents_right)
        
    path_info = {}
    for i in paths:
        node_ids = []  # node ids used in the current node
        # inequality symbols used in the current node
        inequality_symbols = []
        thresholds = []  # thretholds used in the current node
        features = []  # features used in the current node
        parents_left, parents_right = paths[i]
        for idx in range(len(parents_left)):
            if (parents_left[idx] != -1):
                """ the child node is the left child of the parent """
                node_id = parents_left[idx]  # node id
                node_ids.append(node_id)
                inequality_symbols.append(0)
                thresholds.append(threshold[node_id])
                features.append(feature[node_id])
            elif (parents_right[idx] != -1):
                """ the child node is the right child of the parent """
                node_id = parents_right[idx]
                node_ids.append(node_id)
                inequality_symbols.append(1)
                thresholds.append(threshold[node_id])
                features.append(feature[node_id])
            path_info[i] = {'node_id': node_ids,
                            'inequality_symbol': inequality_symbols,
                            'threshold': thresholds,
                            'feature': features}
    return path_info

def esatisfactory_instance(x, epsilon, path_info):
    """
    return the epsilon satisfactory instance of x.
    """
    esatisfactory = copy.deepcopy(x)
    for i in range(len(path_info['feature'])):
        # feature index
        feature_idx = path_info['feature'][i]
        # threshold used in the current node
        threshold_value = path_info['threshold'][i]
        # inequality symbol
        inequality_symbol = path_info['inequality_symbol'][i]
        if inequality_symbol == 0:
            esatisfactory[feature_idx] = threshold_value - epsilon
        elif inequality_symbol == 1:
            esatisfactory[feature_idx] = threshold_value + epsilon
        else:
            print('something wrong')
    return esatisfactory
 
def feature_tweaking(ensemble_classifier, x, class_labels, aim_label, epsilon, cost_func):
    """
    This function return the active feature tweaking vector.
    x: feature vector
    class_labels: list containing the all class labels
    aim_label: the label which we want to transform the label of x to
    """
    """ initialize """
    x_out = copy.deepcopy(x)  # initialize output
    delta_mini = 10**3  # initialize cost
    for estimator in ensemble_classifier:
        if (ensemble_classifier.predict(x.reshape(1, -1)) == estimator.predict(x.reshape(1, -1))
            and estimator.predict(x.reshape(1, -1) != aim_label)):
            paths_info = search_path(estimator, class_labels, aim_label)
            for key in paths_info:
                """ generate epsilon-satisfactory instance """
                path_info = paths_info[key]
                es_instance = esatisfactory_instance(x, epsilon, path_info)
                # if estimator.predict(es_instance.reshape(1, -1)) == aim_label:
                if ensemble_classifier.predict(es_instance.reshape(1, -1)) == aim_label:
                    if cost_func(x, es_instance) < delta_mini:
                        x_out = es_instance
                        delta_mini = cost_func(x, es_instance)
            else:
                continue
    return x_out

# Feature Tweaking method (Based on Tolomei, found in: https://github.com/upura/featureTweakPy) (ensemble_classifier, x, class_labels, aim_label, epsilon, cost_func)
# Added by Anonymous Author
def feat_tweak(x,rf_model,epsilon):
    """
    Function that calls the feature tweaking algorithm and returns the FT counterfactual with respect to instance of interest x
    Input x: Instance of interest
    Input x_label: Label of instance of interest x
    Input rf_model: Random Forest model fitted with the training dataset
    Input epsilon: The rate of change with respect to standard deviation of the features when tweaked.
    Output ft_cf: Feature Tweaking counterfactual to the instance of interest x
    """
    start_time = time.time()
    x_pred = rf_model.predict(x.reshape(1,-1))
    classes = [0,1]
    if x_pred == 0:
        aim = 1
    else:
        aim = 0
    ft_cf = feature_tweaking(rf_model,x,classes,aim,epsilon,euclidean)
    end_time = time.time()
    ft_time = end_time - start_time
    return ft_cf, ft_time
"""
Nearest Approximate Justifier Algorithm (NAJ)
"""

"""
Imports
"""

import numpy as np
from scipy.spatial import distance_matrix
from itertools import permutations

def verify_diff_label(label,model,v):
    """
    Function that verifies if the label given does not match that of the model predicted on instance v
    Input label: Label to be compared
    Input model: Model to be used on instance v
    Input v: Instance to be checked on same label
    Output different: Boolean indicating whether labels are the same or not
    """
    different = False
    v = v.reshape(1, -1)
    label_v = model.predict(v)
    if label_v != label:
        different = True
    return different

def verify_same_label(label,model,v):
    """
    Function that verifies if the label given matches that of the model predicted on instance v
    Input label: Label to be compared
    Input model: Model to be used on instance v
    Input v: Instance to be checked on same label
    Output same: Boolean indicating whether labels are the same or not
    """
    same = True
    v = v.reshape(1, -1)
    label_v = model.predict(v)
    if label_v != label:
        same = False
    return same

def permutation_verify(x,vector,perm,label,model):
    """
    Auxiliary method that verifies a single permutation for binary features
    Input x: Instance of interest
    Input vector: vector of movement between x and target instance
    Input perm: Permutation to test
    Input model: Prediction model used
    Input label: Label of the instance of interest
    Output fail: Whether the permutation could not be verified in terms of the label
    """
    fail = 0
    v = np.copy(x)
    if type(perm) == list:
        for j in perm:
            v[j] += vector[j]
            if verify_diff_label(label,model,v):
                fail = 1
                break
    else:
        v[perm] += vector[perm]
        if verify_diff_label(label,model,v):
            fail = 1
    return fail

def verify_justification(evaluator,model,data):
    """
    Function that outputs whether the instance cf is justified by instance nn_cf or other training instance, with the model as input
    Input evaluator: Evaluator object
    Input model: Trained model to verify justification (connectedness) between nn_cf and cf through the feature space
    Input data: Dataset object
    Output justifier_instance: Instance that justifies cf
    Output justified: 1 or 0, if the instance cf is justified by t or not respectively
    """

    def binary_justification():
        """
        Function that initially verifies binary feature justification
        Output bin_justified: > 0 if the instance x is justified in its binary features by t or not respectively for any binary path
        """

        def recursive_bin_perm(possible_perm):
            """
            Recursively obtain only permutations for which the binary justification is successful
            Output possible_perm: The list of possible permutations of binary indexes that lead to a binary justified instance of interest
            """
            if len(possible_perm) > 0:
                return possible_perm
            bin_diff_index_new = [i for i in bin_diff_index if i not in previous_perm]
            if len(bin_diff_index_new) == 0:
                possible_perm.append(previous_perm)
                return possible_perm
            if len(previous_perm) > 0:
                for i in range(len(bin_diff_index_new)):
                    if len(previous_perm) == 1:
                        add = [previous_perm[0],bin_diff_index_new[i]]
                    else:
                        add = previous_perm.copy()
                        add.extend([bin_diff_index_new[i]]) 
                    bin_diff_index_new[i] = add
            for i in bin_diff_index_new:
                fail = permutation_verify(evaluator.cf,vector,i,evaluator.cf_label,model)
                if fail == 0:
                    if not isinstance(i,list):
                        i = [i]
                    possible_perm = recursive_bin_perm()
            return possible_perm

        def list_bin_perm():
            """
            Obtains only permutations for which the binary justification is successful
            Output possible_perm: The list of possible permutations of binary indexes that lead to a binary justified instance of interest
            """
            perm_list = list(permutations(bin_diff_index,len(bin_diff_index)))
            for i in perm_list:
                if len(possible_perm) > 0:
                    return possible_perm
                v = np.copy(evaluator.cf)
                count = 0
                for j in i:
                    v[j] += vector[j]
                    if verify_same_label(evaluator.cf_label,model,v):
                        count += 1
                    else:
                        break
                if count == len(bin_diff_index):
                    possible_perm.append(i)
            return possible_perm

        vector = evaluator.nn_to_cf - evaluator.cf
        bin_diff_index = np.where((vector != 0) & (data.feat_type == 'bin') & (data.feat_mutable == 1))[0].tolist()
        previous_perm = []
        possible_perm = []
        # if len(bin_diff_index) > 11: # In case there are many permutations, a recursive process is executed to prune and reduce the amount of permutations verified
        #     possible_perm = recursive_bin_perm(possible_perm)
        # else:
        possible_perm = list_bin_perm()
        bin_justified = 1 if len(possible_perm) > 0 else 0
        return bin_justified

    def ordinal_justification():
        """
        Method to verify justification property in ordinal features
        Output ordinal_justified: > 0 if the instance x is justified in its ordinal features by t or not respectively for any ordinal path
        """

        def list_ord_perm():
            """
            Obtains only permutations for which the ordinal justification is successful
            Output possible_perm: The list of possible permutations of ordinal indexes that lead to a ordinal justified instance of interest
            """
            perm_list = list(permutations(ord_diff_index,len(ord_diff_index)))
            for i in perm_list:
                if len(possible_perm) > 0:
                    return possible_perm
                v = np.copy(evaluator.cf)
                count = 0
                for j in i:
                    unviable = False
                    if unviable:
                        break
                    direc_j, step_j = np.sign(vector[j]), data.feat_step.iloc[j]
                    not_close = True
                    while not_close:
                        v[j] += direc_j*step_j
                        if verify_same_label(evaluator.cf_label,model,v):
                            if np.isclose(np.abs(evaluator.nn_to_cf[j] - v[j]),0,rtol=0.000001) or np.sign(evaluator.nn_to_cf[j] - v[j]) != direc_j:
                                v[j] = evaluator.nn_to_cf[j]
                                count += 1
                                not_close = False
                        else:
                            unviable = True
                            break
                if count == len(i):
                    possible_perm.append(i)
            return possible_perm

        vector = evaluator.nn_to_cf - evaluator.cf
        ord_diff_index = np.where((vector != 0) & (data.feat_type == 'num-ord') & (data.feat_mutable == 1))[0].tolist()
        possible_perm = []
        possible_perm = list_ord_perm()
        ord_justified = 1 if len(possible_perm) > 0 else 0
        return ord_justified

    def continuous_justification():
        """
        Function that initially verifies continuous feature justification
        Output justifier_instance: Instance that justifies x
        Output num_justified: 1 or 0, if the instance x is justified in its continuous features by t or not respectively
        """

        def continuous_feat_params():
            """
            Function that outputs parameters needed for continuous feature justification verifying
            Output dist_matrix: Distance matrix among all instances
            Output all_instances: All the instances (np.vstack((x_cont,t,gen_instances)))
            Output label_all_instances: Label of all instances in the matrix
            Output type_all_instances: vector indicating whether the instance is x (x), t (t), from generated instances (g), or from data (d)
            Output epsilon_scan: Distance to check around each instance in the matrix.
            """

            def find_data_equal_feat():
                """
                Function that finds data points in dataset which have same binary feature values as t
                Output data_bin_ord_equal: Dataset containing only instances that have equal value in binary features between the instance of interest and the training instance
                Output data_bin_ord_equal_label: Label of the instances in data_equal_bin
                """
                data_bin_ord_equal = []
                data_bin_ord_equal_label = []
                data_train_set = np.copy(data.jce_train_np)
                for i in range(len(data_train_set)):
                    counter = 0
                    for j in bin_ord_nonmut_index:
                        if data_train_set[i,j] == evaluator.nn_to_cf[j]:
                            counter += 1
                    if counter == len(bin_ord_nonmut_index) and not (evaluator.nn_to_cf == data_train_set[i]).all():
                        data_bin_ord_equal.append(data_train_set[i])
                        data_bin_ord_equal_label.append(data.train_target[i])
                return np.array(data_bin_ord_equal), np.array(data_bin_ord_equal_label)

            lower = [0]*len(evaluator.nn_to_cf)
            upper = [1]*len(evaluator.nn_to_cf)
            cf_cont = np.copy(evaluator.cf)
            bin_ord_nonmut_index = np.where((data.feat_type != 'num-con') | (data.feat_mutable == 0))[0].tolist()
            for i in bin_ord_nonmut_index:
                lower[i] = evaluator.nn_to_cf[i]
                upper[i] = evaluator.nn_to_cf[i]
                cf_cont[i] = evaluator.nn_to_cf[i]
            gen_instances = np.random.uniform(lower,upper,size=(evaluator.n_feat*len(evaluator.nn_to_cf),len(evaluator.nn_to_cf)))
            label_gen_instances = model.predict(gen_instances)
            data_bin_ord_equal, data_bin_ord_equal_label = find_data_equal_feat()
            if data_bin_ord_equal.shape[0] == 0:
                all_instances = np.vstack((cf_cont,evaluator.nn_to_cf,gen_instances))
                label_all_instances = np.hstack((evaluator.cf_label,evaluator.cf_label,label_gen_instances))
                type_all_instances = ['x']+['t']+['g']*len(gen_instances)    
            else:
                all_instances = np.vstack((cf_cont,data_bin_ord_equal,evaluator.nn_to_cf,gen_instances))
                label_all_instances = np.hstack((evaluator.cf_label,data_bin_ord_equal_label,evaluator.cf_label,label_gen_instances))
                type_all_instances = ['x']+['d']*len(data_bin_ord_equal)+['t']+['g']*len(gen_instances)
            dist_matrix = distance_matrix(all_instances,all_instances)
            count_cont = np.sum(data.feat_type != 'bin')
            epsilon_scan = np.sqrt(count_cont)/5 # This value may be changed for some other value of interest (this was chosen for the results of the r radius study)
            return dist_matrix, all_instances, label_all_instances, type_all_instances, epsilon_scan

        def chain(index_list_checked,index_prev,index_next,num_justified):
            """
            Function that creates a chain of paths and finds whether there is a continuous path between the instances. Uses Depth-First search
            Input x: closest CF to the instance of interest and to be verified for justification with the chain
            Input index_list_checked: Set of tuples of interconnected instances
            Input index_prev: instance previously checked
            Input index_next: instance to check next
            Output index_list_checked: Set of tuples of interconnected instances
            Output num_justified: Variable that becomes 1 when justification is verified  
            """
            index_list_checked.append((index_prev,index_next))
            index_prev = index_next
            index_close = np.where((dist_matrix[index_next,:] <= epsilon_scan) & (dist_matrix[index_next,:] > 0))[0]
            for j in index_close:
                if len([i for i in index_list_checked if j in i]) > 0:
                    continue
                elif type_all_instances[j] == 'g':
                    if label_all_instances[j] != evaluator.cf_label:
                        index_list_checked.append((index_prev,j,'wrong label'))
                    else:
                        index_next = j
                        index_list_checked, num_justified = chain(index_list_checked,index_prev,index_next,num_justified)    
                        if num_justified == 1:
                            break
                elif type_all_instances[j] == 'd' or type_all_instances[j] == 't':
                    if label_all_instances[j] != evaluator.cf_label:
                        index_list_checked.append((index_prev,j,'wrong label'))
                    else:
                        index_list_checked.append((index_prev,j,'justified',dist_matrix[0,j]))
                        num_justified = 1
                        return index_list_checked, num_justified
            return index_list_checked, num_justified

        dist_matrix, all_instances, label_all_instances, type_all_instances, epsilon_scan = continuous_feat_params()
        index_list_checked = []
        index_prev = -1
        index_next = 0
        num_justified = 0
        justifier_instance = evaluator.cf
        instance_chain, num_justified = chain(index_list_checked,index_prev,index_next,num_justified)
        justifying_tuples = [i for i in instance_chain if 'justified' in i]
        if len(justifying_tuples) > 0:
            justifying_tuples.sort(key=lambda x: x[3])
            closest_justifying_index = justifying_tuples[0][1]
            justifier_instance = all_instances[closest_justifying_index,:]
            num_justified = 1
        return justifier_instance, num_justified

    justifier_instance = evaluator.cf
    justified = 0
    bin_justified = 0
    if np.array_equal(evaluator.cf,evaluator.nn_to_cf):
        justifier_instance = evaluator.nn_to_cf
        justified = 1
    else:
        bin_justified = binary_justification()
        if bin_justified > 0:
            ord_justified = ordinal_justification()
            if ord_justified > 0:
                justifier_instance, num_justified = continuous_justification()
                if num_justified:
                    justified = 1
    return justifier_instance, justified
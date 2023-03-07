"""
Minimum Observable (MO)
"""

"""
Imports
"""
import numpy as np
import time
from evaluator_constructor import distance_calculation

class MO:

    def __init__(self, counterfactual) -> None:
        self.normal_x_cf, self.run_time = min_obs(counterfactual)

def min_obs(counterfactual):
    """
    Function that returns the minimum observable counterfactual with respect to instance of interest x
    """
    data = counterfactual.data
    ioi = counterfactual.ioi
    model = counterfactual.model
    type = counterfactual.type

    start_time = time.time()
    mo_cf = None
    all_data = np.vstack((data.transformed_train_np, data.transformed_test_np))
    all_labels = np.hstack((data.train_target, model.model.predict(data.transformed_test_np)))
    data_distance_mo = []
    for i in range(all_data.shape[0]):
        dist = distance_calculation(all_data[i], ioi.normal_x, data)
        data_distance_mo.append((all_data[i], dist, all_labels[i]))      
    data_distance_mo.sort(key=lambda x: x[1])
    for i in data_distance_mo:
        if i[2] != ioi.label and not np.array_equal(ioi.normal_x, i[0]):
            mo_cf = i[0]
            break
    if mo_cf is None:
        print(f'MO could not find a feasible CF!')
        end_time = time.time()
        return mo_cf, end_time - start_time
    end_time = time.time()
    mo_time = end_time - start_time
    return mo_cf, mo_time
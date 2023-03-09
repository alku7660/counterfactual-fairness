import numpy as np
import pandas as pd
import time
from evaluator_constructor import distance_calculation

def inverse_transform_original(centroid, data):
    """
    DESCRIPTION:            Transforms an centroid to the original features
    
    INPUT:
    centroid:               centroid of interest

    OUTPUT:
    original_centroid_df:   centroid of interest in the original feature format
    """
    centroid_index = centroid.index
    original_centroid_df = pd.DataFrame(index=centroid_index)
    if len(data.bin_enc_cols) > 0:
        centroid_bin = data.bin_enc.inverse_transform(centroid[data.bin_enc_cols])
        centroid_bin_pd = pd.DataFrame(data=centroid_bin, index=centroid_index, columns=data.binary)
        original_centroid_df = pd.concat((original_centroid_df, centroid_bin_pd), axis=1)
    if len(data.cat_enc_cols) > 0:
        centroid_cat = data.cat_enc.inverse_transform(centroid[data.cat_enc_cols])
        centroid_cat_pd = pd.DataFrame(data=centroid_cat, index=centroid_index, columns=data.categorical)
        original_centroid_df = pd.concat((original_centroid_df, centroid_cat_pd), axis=1)
    if len(data.numerical) > 0:
        centroid_num = data.scaler.inverse_transform(centroid[data.numerical])
        centroid_num_pd = pd.DataFrame(data=centroid_num, index=centroid_index, columns=data.numerical)
        original_centroid_df = pd.concat((original_centroid_df, centroid_num_pd), axis=1)
    return original_centroid_df

class CENTROID:

    def __init__(self, centroid_idx, centroid_list, feat_val, feat, data, model, type='euclidean') -> None:
        self.centroid_idx = centroid_idx
        self.feat = feat
        self.feat_val = feat_val
        self.normal_x = centroid_list[centroid_idx]
        self.normal_x_df = pd.DataFrame(data=self.normal_x, index=[self.centroid_idx], columns=data.processed_features)
        self.x = inverse_transform_original(self.normal_x, data)
        self.x_label = model.model.predict(self.normal_x.reshape(1, -1))
        self.train_sorted, self.train_sorting_time = self.sorted(data, type)
    
    def sorted(self, data, type):
        """
        Function to organize dataset with respect to distance to instance x
        """
        start_time = time.time()
        sort_data_distance = []
        for i in range(data.transformed_train_np.shape[0]):
            dist = distance_calculation(data.transformed_train_np[i], self.normal_centroid, data, type)
            sort_data_distance.append((data.transformed_train_np[i], dist, data.train_target[i]))      
        sort_data_distance.sort(key=lambda x: x[1])
        end_time = time.time()
        total_time = end_time - start_time 
        return sort_data_distance, total_time
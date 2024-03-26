import pandas as pd

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

def estimate_sensitive_group_positive(data, feat, feat_val):
    """
    Estimates the amount of ground truth positives in the feature sensitive group given as parameter
    """
    sensitive_group_df = data.test_df.loc[(data.test_df[feat] == feat_val) & (data.test_target == data.desired_class)]
    return len(sensitive_group_df)

class Centroid:

    def __init__(self, centroid_idx, centroid_list, cluster_size, feat_val, feat, data, model) -> None:
        self.centroid_idx = centroid_idx
        self.cluster_size = cluster_size
        self.feat = feat
        self.feat_val = feat_val
        self.normal_x_df = centroid_list[0]
        self.normal_x = self.normal_x_df.values[0]
        self.x = inverse_transform_original(self.normal_x_df, data).values
        self.x_label = model.model.predict(self.normal_x.reshape(1, -1))
        self.positives_sensitive_group = estimate_sensitive_group_positive(data, feat, feat_val)
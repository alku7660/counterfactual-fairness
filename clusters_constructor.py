import numpy as np
import pandas as pd
import copy
from sklearn.cluster import AgglomerativeClustering
from centroid_constructor import Centroid

class Clusters:
    def __init__(self, data, model, metric='single') -> None:
        self.false_undesired_test_df = data.false_undesired_test_df
        self.transformed_false_undesired_test_df = data.transformed_false_undesired_test_df
        self.num_instances_false_undesired_test = len(self.false_undesired_test_df)
        self.feat_protected = data.feat_protected
        self.undesired_class = data.undesired_class
        self.feat_type = data.feat_type
        self.feat_cat = data.feat_cat
        self.metric = metric
        self.sensitive_feat_idx_dict = self.select_instances_by_sensitive_group()
        self.clusters, self.centroids = self.find_viable_clusters(model)
        self.centroids = self.define_centroids(data, model)

    def select_instances_by_sensitive_group(self):
        """
        Obtains indices of each sensitive group and stores them in a dict
        """
        sensitive_feat_idx_dict = dict()
        for key in self.feat_protected.keys():
            idx_list_by_sensitive_group_dict = dict()
            value_dict = self.feat_protected[key]
            for value in value_dict.keys():
                sensitive_group_df = self.false_undesired_test_df.loc[self.false_undesired_test_df[key] == value]
                idx_list_sensitive_group = sensitive_group_df.index.to_list()
                idx_list_by_sensitive_group_dict[value] = idx_list_sensitive_group
            sensitive_feat_idx_dict[key] = idx_list_by_sensitive_group_dict
        return sensitive_feat_idx_dict
    
    def calculate_centroid(self, instances_df, method='mode'):
        """
        Finds the centroid of the given set of instances
        """
        if method == 'mean':
            centroid = instances_df.mean(axis=0).to_frame().T
        elif method == 'mode':
            centroid = copy.deepcopy(instances_df.iloc[0])
            feat_checked = []
            for feat in self.feat_type.index.to_list():
                if feat not in feat_checked:
                    if self.feat_type.loc[feat] in ['bin','ord']:
                        centroid[feat] = instances_df[feat].mode()[0]
                    elif self.feat_type.loc[feat] == 'cat':
                        feat_cat = self.feat_cat.loc[feat]
                        feat_cat_cols = [i for i in self.feat_type.index if self.feat_cat.loc[i] == feat_cat]
                        feat_cat_col_mode = feat_cat_cols[instances_df[feat_cat_cols].sum().argmax()]
                        centroid[feat_cat_cols] = [0]*len(feat_cat_cols)
                        centroid[feat_cat_col_mode] = 1
                        feat_checked.extend(feat_cat_cols)
                    else:
                        centroid[feat] = instances_df[feat].mean()
        return centroid.to_frame().T

    def find_viable_clusters(self, model):
        """
        Finds the appropriate clusters so that the predicted label is still the negative class
        """

        def hierarchical_clusters(instances_df, clusters_instances_list=[], clusters_centroids_list=[]):
            """
            If centroid not of undesired class, start hierarchical clustering
            """
            clustering = AgglomerativeClustering(n_clusters = 2, linkage=self.metric)
            clustering.fit(instances_df)
            instances_df['cluster'] = clustering.labels_
            unique_clustering_labels = np.unique(clustering.labels_)
            for j in unique_clustering_labels:
                cluster_instances_df = instances_df.loc[instances_df['cluster'] == j]
                cluster_instances_df_j_cluster = copy.deepcopy(cluster_instances_df)
                del cluster_instances_df_j_cluster['cluster']
                centroid = self.calculate_centroid(cluster_instances_df_j_cluster)
                centroid_label = model.model.predict(centroid)
                if centroid_label != self.undesired_class:
                    clusters_instances_list, clusters_centroids_list = hierarchical_clusters(cluster_instances_df_j_cluster, clusters_instances_list, clusters_centroids_list)
                else:
                    clusters_instances_list.append(cluster_instances_df_j_cluster.index)
                    clusters_centroids_list.append(centroid)
            return clusters_instances_list, clusters_centroids_list

        feature_cluster_instances_dict = dict()
        feature_cluster_centroids_dict = dict()
        for feat_name in self.sensitive_feat_idx_dict.keys():
            idx_list_by_sensitive_group = self.sensitive_feat_idx_dict[feat_name]
            sensitive_group_cluster_instances_dict = dict()
            sensitive_group_cluster_centroids_dict = dict()
            for feat_val in idx_list_by_sensitive_group:
                clusters_instances_list, clusters_centroids_list = [], []
                idx_list_feat_val = idx_list_by_sensitive_group[feat_val]
                sensitive_group_instances_df = self.transformed_false_undesired_test_df.loc[idx_list_feat_val]
                centroid = self.calculate_centroid(sensitive_group_instances_df)
                centroid_label = model.model.predict(centroid)
                if centroid_label != self.undesired_class:
                    clusters_instances_list, clusters_centroids_list = hierarchical_clusters(sensitive_group_instances_df, clusters_instances_list=clusters_instances_list, clusters_centroids_list=clusters_centroids_list)
                else:
                    clusters_instances_list, clusters_centroids_list = [idx_list_feat_val], [centroid]
                sensitive_group_cluster_instances_dict[feat_val] = clusters_instances_list
                sensitive_group_cluster_centroids_dict[feat_val] = clusters_centroids_list
            feature_cluster_instances_dict[feat_name] = sensitive_group_cluster_instances_dict
            feature_cluster_centroids_dict[feat_name] = sensitive_group_cluster_centroids_dict
        return feature_cluster_instances_dict, feature_cluster_centroids_dict

    def define_centroids(self, data, model):
        """
        Creates the list of centroid objects
        """
        centroid_list = []
        centroids_feat_list = list(self.centroids.keys())
        for feat in centroids_feat_list:
            feat_val_list = list(centroids_feat_list[feat].keys())
            for feat_val in feat_val_list:
                centroid_list = self.centroids[feat][feat_val]
                for centroid_idx in range(len(centroid_list)):
                    centroid = Centroid(centroid_idx, centroid_list, feat_val, feat, data, model, type='euclidean')
                    centroid_list.append(centroid)
        return centroid_list

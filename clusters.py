import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

class CLUSTERS:
    def __init__(self, data, model, metric='single') -> None:
        self.false_undesired_test_df = data.false_undesired_test_df
        self.transformed_false_undesired_test_df = data.transformed_false_undesired_test_df
        self.num_instances_false_undesired_test = len(self.false_undesired_test_df)
        self.feat_protected = data.feat_protected
        self.undesired_class = data.undesired_class
        self.metric = metric
        self.sensitive_feat_idx_dict = self.select_instances_by_sensitive_group()
        self.clusters, self.centroids = self.find_viable_clusters(model)

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
    
    def calculate_centroid(self, instances_df, method='mean'):
        """
        Finds the centroid of the given set of instances
        """
        if method == 'mean':
            centroid = instances_df.mean(axis=0).to_frame().T
        elif method == 'max':
            centroid = instances_df.max(axis=0).to_frame().T
        return centroid

    def find_viable_clusters(self, model):
        """
        Finds the appropriate clusters so that the predicted label is still the negative class
        """

        def hierarchical_clusters(instances_df):
            """
            If centroid not of undesired class, start hierarchical clustering
            """
            for i in range(2, len(instances_df)+1):
                clustering = AgglomerativeClustering(n_clusters = i, linkage=self.metric)
                clustering.fit(instances_df)
                instances_df['cluster'] = clustering.labels_
                unique_clustering_labels = np.unique(clustering.labels_)
                clusters_instances_list = []
                clusters_centroids_list = []
                for j in unique_clustering_labels:
                    cluster_instances_df = instances_df.loc[instances_df['cluster'] == j]
                    centroid = self.calculate_centroid(cluster_instances_df)
                    centroid_label = model.model.predict(centroid)
                    if centroid_label != self.undesired_class:
                        break
                    else:
                        clusters_instances_list.append(cluster_instances_df.index)
                        clusters_centroids_list.append(centroid)
                if len(clusters_instances_list) == len(unique_clustering_labels):
                    break
            return clusters_instances_list, clusters_centroids_list

        feature_cluster_instances_dict = dict()
        feature_cluster_centroids_dict = dict()
        for feat_name in self.sensitive_feat_idx_dict.keys():
            idx_list_by_sensitive_group = self.sensitive_feat_idx_dict[feat_name]
            sensitive_group_cluster_instances_dict = dict()
            sensitive_group_cluster_centroids_dict = dict()
            for feat_val in idx_list_by_sensitive_group:
                idx_list_feat_val = idx_list_by_sensitive_group[feat_val]
                sensitive_group_instances_df = self.transformed_false_undesired_test_df.iloc[idx_list_feat_val]
                centroid = self.calculate_centroid(sensitive_group_instances_df, method='mean')
                centroid_label = model.model.predict(centroid)
                if centroid_label != self.undesired_class:
                    clusters_instances_list, clusters_centroids_list = hierarchical_clusters(sensitive_group_instances_df)
                else:
                    clusters_instances_list, clusters_centroids_list = [idx_list_feat_val], [centroid]
                sensitive_group_cluster_instances_dict[feat_val] = clusters_instances_list
                sensitive_group_cluster_centroids_dict[feat_val] = clusters_centroids_list
            feature_cluster_instances_dict[feat_name] = sensitive_group_cluster_instances_dict
            feature_cluster_centroids_dict[feat_name] = feature_cluster_centroids_dict
        return feature_cluster_instances_dict, feature_cluster_centroids_dict
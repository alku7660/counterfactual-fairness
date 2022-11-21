from carla import Data, MLModel
import pandas as pd
import numpy as np

 # This file is adapted from the CARLA framework. The CARLA framework is meant 
 # for Counterfactual Explanation method applications. Please, see the follwing links for further information:
 # ArXiv paper:   https://arxiv.org/pdf/2108.00783.pdf
 # GitHub repo:   https://github.com/carla-recourse/CARLA
 # Documentation: https://carla-counterfactual-and-recourse-library.readthedocs.io/en/latest/

 # Custom data set implementations need to inherit from the Data interface
class MyOwnDataSet(Data):
    def __init__(self, data_obj):
        # The data set can e.g. be loaded in the constructor
        # self._dataset = data_obj.raw_df
        # self._dataset_train = data_obj.train_df
        # self._dataset_test = data_obj.test_df
        # self._name = data_obj.name
        # self._categorical = data_obj.carla_categorical
        # self._continuous = data_obj.carla_continuous
        # self.encoder = data_obj.carla_enc
        # self.scaler = data_obj.carla_scaler
        # self.processed_features = self._continuous + list(self.encoder.get_feature_names(self._categorical))

        self._dataset_train = data_obj.carla_transformed_train_df
        self._dataset_test = data_obj.carla_transformed_test_df
        self._dataset = pd.concat((self._dataset_train, self._dataset_test), axis=0)
        self._dataset[data_obj.label_str] = pd.DataFrame(np.concatenate((data_obj.train_target, data_obj.test_target)), index=list(self._dataset.index))
        self._name = data_obj.name
        self._categorical = data_obj.carla_categorical
        self._continuous = data_obj.carla_continuous
        self.encoder = data_obj.carla_enc
        self.scaler = data_obj.carla_scaler
        self._enc_categorical = list(self.encoder.get_feature_names(self._categorical))
        self.processed_features = self._continuous + self._enc_categorical

    # List of all categorical features
    @property
    def categorical(self):
        return self._categorical

    # List of all continous features
    @property
    def continuous(self):
        return self._continuous

    # List of all immutable features which
    # should not be changed by the recourse method
    @property
    def immutables(self):
        if self._name == 'adult':
            immutables = ['Sex','Race','AgeGroup']
        elif self._name == 'kdd_census' or self._name == 'compass' or self._name == 'law':
            immutables = ['Sex','Race']
        elif self._name == 'german' or self._name == 'dutch' or self._name == 'diabetes' or self._name == 'oulad':
            immutables = ['Sex']
        elif self._name == 'bank':
            immutables = ['AgeGroup','MaritalStatus']
        elif self._name == 'credit':
            immutables = ['isMale','isMarried','EducationLevel']
        elif self._name == 'student':
            immutables = ['Sex','AgeGroup']
        return immutables

    # Feature name of the target column
    @property
    def target(self):
        if self._name == 'adult':
            label = 'label'
        elif self._name == 'kdd_census' or self._name == 'german' or self._name == 'diabetes':
            label = 'Label'
        elif self._name == 'dutch':
            label = 'Occupation'
        elif self._name == 'bank':
            label = 'Subscribed'
        elif self._name == 'credit':
            label = 'NoDefaultNextMonth (label)'
        elif self._name == 'compass':
            label = 'TwoYearRecid (label)'
        elif self._name == 'german':
            label = 'GoodCustomer (label)'
        elif self._name == 'student' or self._name == 'oulad':
            label = 'Grade'
        elif self._name == 'law':
            label = 'BarExam'
        return label

    # Non-encoded and  non-normalized, raw data set
    @property
    def df(self):
        return self._dataset

    @property
    def df_train(self):
        return self._dataset_train

    @property
    def df_test(self):
        return self._dataset_test

    def transform(self, df):
        df_cat = pd.DataFrame(self._mlmodel.data.encoder.transform(df[self._mlmodel.data.categorical]), index=df.index, columns=list(self.encoder.get_feature_names(self._categorical)))
        df_con = pd.DataFrame(self._mlmodel.data.scaler.transform(df[self._mlmodel.data.continuous]), index=df.index, columns=self._mlmodel.data.continuous)
        transformed_df = pd.concat((df_con, df_cat), axis=1)
        return transformed_df
    
    def inverse_transform(self, df):
        df_cat = pd.DataFrame(self.encoder.inverse_transform(df[list(self.encoder.get_feature_names(self._categorical))]), index=df.index, columns=self._categorical)
        df_con = pd.DataFrame(self.scaler.inverse_transform(df[self._continuous]), index=df.index, columns=self._continuous)
        original_df = pd.concat((df_con, df_cat), axis=1)
        return original_df

 # Custom black-box models need to inherit from
 # the MLModel interface
class MyOwnModel(MLModel):
    def __init__(self, carla_data, data, model):
        super().__init__(carla_data)
        # The constructor can be used to load or build an
        # arbitrary black-box-model
        self._mymodel = model.carla_sel
        self.undesired_class = data.undesired_class
        
        # Define a fitted sklearn scaler to normalize input data
        self.scaler = data.carla_scaler

        # Define a fitted sklearn encoder for binary input data
        self.encoder = data.carla_enc
        self.features = carla_data.processed_features
        # self.features = carla_data.processed_features

    # List of the feature order the ml model was trained on
    @property
    def feature_input_order(self):
        return self.features

    # The ML framework the model was trained on
    @property
    def backend(self):
        return "sklearn"

    # The black-box model object
    @property
    def raw_model(self):
        return self._mymodel

    # The predict function outputs
    # the continous prediction of the model
    def predict(self, x):
        return self._mymodel.predict(x)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        return self._mymodel.predict_proba(x)
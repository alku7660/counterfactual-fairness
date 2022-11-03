"""
Dataset loader
"""

"""
Imports
"""

import os
import copy
import pickle
from model_params import clf_model, best_model_params
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from support import dataset_dir

def euclidean(x1,x2):
    """
    DESCRIPTION:    Calculation of the euclidean distance between two different instances
    
    INPUT:
    x1:             Instance 1
    x2:             Instance 2
    
    OUTPUT:
    distance:       Euclidean distance between x1 and x2
    """
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

def sort_data_distance(x,data,data_label):
    """
    DESCRIPTION:    Function to organize dataset with respect to distance to instance x
    
    INPUT:
    x:              Instance (can be the instance of interest or a synthetic instance)
    data:           Training dataset (Numpy array format)
    data_label:     Training dataset label (Numpy array format)
    
    OUTPUT:
    data_sorted_distance: Training dataset sorted by distance w.r.t. the instance of interest x
    """
    sort_data_distance = []
    for i in range(len(data)):
        dist = euclidean(data[i],x)
        sort_data_distance.append((data[i],dist,data_label[i]))      
    sort_data_distance.sort(key=lambda x: x[1])
    return sort_data_distance

class Dataset:
    """
    **Parts of this class are adapted from the MACE algorithm methodology (please, see: https://github.com/amirhk/mace)**
    DESCRIPTION:    Dataset Class

    INPUT:          
    seed_int:           Seed integer number
    train_fraction:     Fraction of the dataset to be used for training and validation
    data_str:           Name of the dataset to be loaded
    label_str:          Name of the column corresponding to the label of the dataset
    raw_df:             Unprocessed dataset
    raw_df_cols:        List of original names of the features (before preprocessing)
    binary:             Original binary feature names
    categorical:        Original categorical feature names
    numerical:          Original numerical feature names (ordinal and continuous)
    step:               Step size to be used in continuous feature discretization
    carla_categorical:  Categorial feature names used in the CARLA framework
    carla_continuous:   Continuous feature names used in the CARLA framework
    """

    def __init__(self, seed_int, train_fraction, data_str, label_str,
                 raw_df, binary, categorical, numerical, step,
                 carla_categorical, carla_continuous):

        self.seed = seed_int
        self.train_fraction = train_fraction
        self.name = data_str
        self.label_str = label_str
        self.raw_df = raw_df
        self.raw_df_cols = raw_df.columns
        self.binary = binary
        self.categorical = categorical
        self.numerical = numerical
        self.bin_enc = None
        self.step = step
        self.carla_categorical = carla_categorical
        self.carla_continuous = carla_continuous
        self.train_df, self.test_df, self.train_target, self.test_target = train_test_split(self.raw_df,self.raw_df[self.label_str],train_size=self.train_fraction,random_state=self.seed)
        self.train_df, self.train_target, self.test_df = self.data_balancing_target_filter() 
        self.bin_enc, self.cat_enc, self.scaler, self.bin_enc_cols, self.cat_enc_cols = self.encoder_scaler_fit()
        self.transformed_train_df, self.transformed_train_np, self.transformed_cols = self.transform_train()
        self.transformed_test_df, self.transformed_test_np = self.transform_test(self.test_df)
        self.carla_enc, self.carla_scaler, self.carla_enc_cols = self.carla_encoder_scaler_fit()
        self.carla_transformed_train_df, self.carla_transformed_test_df, self.carla_transformed_cols = self.carla_transform_train_test()
        self.undesired_class = self.undesired_class_data()
        self.feat_type = self.define_feat_type()
        self.feat_protected = self.define_protected()
        self.feat_mutable = self.define_mutability()
        self.feat_dir = self.define_directionality()
        self.feat_cost = self.define_feat_cost()
        self.feat_step = self.define_feat_step()
        self.feat_cat = self.define_category_groups()
        self.train_sorted = None

    def data_balancing_target_filter(self):
        """
        DESCRIPTION:    Method that balances the training dataset (Adapted from MACE algorithm methodology - please see: https://github.com/amirhk/mace)
        
        INPUT:
        self
        
        OUTPUT:
        train_df:       Training dataset DataFrame
        train_target:   Training target column
        test_df:        Testing dataset DataFrame 
        """
        unique_values_and_count = self.train_target.value_counts()
        number_of_subsamples_per_class = unique_values_and_count.min() // 10 * 10
        train_df = pd.concat([self.train_df[(self.train_df[self.label_str] == 0).to_numpy()].sample(number_of_subsamples_per_class, random_state = self.seed),
        self.train_df[(self.train_df[self.label_str] == 1).to_numpy()].sample(number_of_subsamples_per_class, random_state = self.seed),]).sample(frac = 1, random_state = self.seed)
        train_target = self.train_df[self.label_str]
        test_df = self.test_df.copy()
        del train_df[self.label_str[0]]
        del test_df[self.label_str[0]]
        return train_df, train_target, test_df

    def encoder_scaler_fit(self):
        """
        DESCRIPTION:    Method that fits the encoder and scaler for the dataset
        
        INPUT:
        self

        OUTPUT:
        bin_enc:        Fitted binary encoder
        cat_enc:        Fitted categorical encoder
        scaler:         Fitted scaler
        bin_enc_cols:   Binary encoded feature names
        cat_enc_cols:   Categorical encoded feature names
        """
        bin_enc = OneHotEncoder(drop='if_binary',dtype=np.uint8,handle_unknown='ignore')
        cat_enc = OneHotEncoder(drop='if_binary',dtype=np.uint8,handle_unknown='ignore')
        scaler = MinMaxScaler(clip=True)
        train_data_bin, train_data_cat, train_data_num = self.train_df[self.binary], self.train_df[self.categorical], self.train_df[self.numerical]
        bin_enc.fit(train_data_bin)
        cat_enc.fit(train_data_cat)
        scaler.fit(train_data_num)
        bin_enc_cols = bin_enc.get_feature_names_out(self.binary)
        cat_enc_cols = cat_enc.get_feature_names_out(self.categorical)
        return bin_enc, cat_enc, scaler, bin_enc_cols, cat_enc_cols

    def transform_train(self):
        """
        DESCRIPTION:            Method that fits the encoder and scaler for the dataset and processes the training dataset

        INPUT:
        self

        OUTPUT:
        transformed_train_df:   Transformed training dataset DataFrame
        transformed_train_np:   Transformed training dataset Numpy array
        transformed_cols:       Columns of the transformed dataset
        """
        train_data_bin, train_data_cat, train_data_num = self.train_df[self.binary], self.train_df[self.categorical], self.train_df[self.numerical]
        enc_train_data_bin = self.bin_enc.transform(train_data_bin).toarray()
        enc_train_data_cat = self.cat_enc.transform(train_data_cat).toarray()
        scaled_train_data_num = self.scaler.transform(train_data_num)
        scaled_train_data_num_df = pd.DataFrame(scaled_train_data_num, index=train_data_num.index, columns=self.numerical)
        enc_train_data_bin_df = pd.DataFrame(enc_train_data_bin, index=train_data_bin.index, columns=self.bin_enc_cols)
        enc_train_data_cat_df = pd.DataFrame(enc_train_data_cat, index=train_data_cat.index, columns=self.oh_cat_enc_cols)
        transformed_train_df = pd.concat((enc_train_data_bin_df, enc_train_data_cat_df, scaled_train_data_num_df),axis=1)
        transformed_cols = self.transformed_train_df.columns.to_list()
        transformed_train_np = self.transformed_train_df.to_numpy()
        return transformed_train_df, transformed_train_np, transformed_cols

    def carla_encoder_scaler_fit(self):
        """
        DESCRIPTION:        Method that fits the encoder and scaler for the dataset and transforms the training dataset according to the CARLA framework

        INPUT:
        self

        OUTPUT:
        carla_enc:          Fitted encoder used for the CARLA framework
        carla_scaler:       Fitted scaler for the CARLA framework
        carla_enc_cols:     CARLA framework encoder columns 
        """
        carla_enc = OneHotEncoder(drop='if_binary',dtype=np.uint8,handle_unknown='ignore',sparse=False)
        carla_scaler = MinMaxScaler(clip=True)
        carla_train_data_cat, carla_train_data_cont = self.train_df[self.carla_categorical], self.train_df[self.carla_continuous]
        carla_enc.fit(carla_train_data_cat)
        carla_scaler.fit(carla_train_data_cont)
        carla_enc_cols = self.carla_enc.get_feature_names_out(self.carla_categorical)
        return carla_enc, carla_scaler, carla_enc_cols

    def carla_transform_train_test(self):
        """
        DESCRIPTION:                Method that fits the encoder and scaler for the dataset and transforms the training dataset according to the CARLA framework

        INPUT:
        self

        OUTPUT:
        carla_transformed_train_df: CARLA transformed training dataset (DataFrame)
        carla_transformed_test_df:  CARLA transformed testing dataset (DataFrame)
        carla_transformed_cols:     CARLA transformed feature names
        """
        carla_train_data_cat, carla_train_data_cont = self.train_df[self.carla_categorical], self.train_df[self.carla_continuous]
        enc_carla_train_data_cat = self.carla_enc.transform(carla_train_data_cat)
        scaled_carla_train_data_cont = self.carla_scaler.transform(carla_train_data_cont)
        enc_carla_train_data_cat_df = pd.DataFrame(enc_carla_train_data_cat, index=carla_train_data_cat.index, columns=self.carla_enc_cols)
        scaled_carla_train_data_cont_df = pd.DataFrame(scaled_carla_train_data_cont, index=carla_train_data_cont.index, columns=self.carla_continuous)
        carla_transformed_train_df = pd.concat((scaled_carla_train_data_cont_df,enc_carla_train_data_cat_df),axis=1)
        carla_transformed_cols = carla_transformed_train_df.columns.to_list()

        carla_test_data_cat, carla_test_data_cont = self.test_df[self.carla_categorical], self.test_df[self.carla_continuous]
        enc_carla_test_data_cat = self.carla_enc.transform(carla_test_data_cat)
        scaled_carla_test_data_cont = self.carla_scaler.transform(carla_test_data_cont)
        enc_carla_test_data_cat_df = pd.DataFrame(enc_carla_test_data_cat, index=carla_test_data_cat.index, columns=self.carla_enc_cols)
        scaled_carla_test_data_cont_df = pd.DataFrame(scaled_carla_test_data_cont, index=carla_test_data_cont.index, columns=self.carla_continuous)
        carla_transformed_test_df = pd.concat((scaled_carla_test_data_cont_df, enc_carla_test_data_cat_df), axis=1)
        return carla_transformed_train_df, carla_transformed_test_df, carla_transformed_cols
        
    def transform_test(self, df):
        """
        DESCRIPTION:                Method that uses the encoder and scaler for the dataset and processes the testing dataset

        INPUT:
        df:                         DataFrame on which to apply the transformation (could be different to the testing dataset)

        OUTPUT:
        df:                         Data DataFrame transformed using the corresponding encoder and scaler
        arr:                        Numpy array version of the DataFrame
        """
        data_bin, data_cat, data_num = df[self.binary], df[self.categorical], df[self.numerical]
        enc_data_bin, enc_data_cat = self.bin_enc.transform(data_bin).toarray(), self.cat_enc.transform(data_cat).toarray()
        scaled_data_num = self.scaler.transform(data_num)
        enc_data_bin_df = pd.DataFrame(enc_data_bin,index=data_bin.index,columns=self.bin_enc_cols)
        enc_data_cat_df = pd.DataFrame(enc_data_cat,index=data_cat.index,columns=self.oh_cat_enc_cols)
        scaled_data_num_df = pd.DataFrame(scaled_data_num,index=data_num.index,columns=self.numerical)
        df = pd.concat((enc_data_bin_df, enc_data_cat_df, scaled_data_num_df),axis=1)
        arr = df.to_numpy()
        return df, arr

    # def transform_to_jce_format(self,num_data,enc_bin_data,enc_cat_data):
    #     """
    #     Method that transforms an instance of interest to a comparable format according to the dataset features
    #     Input num_data: The numerical (continuous) variables in DataFrame transformed
    #     Input enc_bin_data: The binary variables transformed in DataFrame
    #     Input enc_cat_cata: The categorical variables transformed in DataFrame
    #     Output enc_jce_data_df: The instance in pandas DataFrame
    #     """
    #     if self.name == 'adult':
    #         enc_jce_data_df = pd.concat((enc_bin_data[self.bin_enc_cols[0]],num_data[self.numerical[0]],
    #                             enc_bin_data[self.bin_enc_cols[1:3]],num_data[self.numerical[1:5]],
    #                             enc_cat_data[self.oh_jce_cat_enc_cols[:7]],num_data[self.numerical[-1]],
    #                             enc_cat_data[self.oh_jce_cat_enc_cols[7:]]),axis=1)
    #     elif self.name in ['kdd_census','german','dutch','bank','credit','diabetes','student','oulad','law']:
    #         enc_jce_data_df = pd.concat((enc_bin_data,num_data,enc_cat_data),axis=1)
    #     elif self.name == 'credit':
    #         enc_jce_data_df = pd.concat((enc_bin_data[self.bin_enc_cols[:2]],num_data[self.numerical[0:9]],
    #                             enc_bin_data[self.bin_enc_cols[2:]],num_data[self.numerical[9:]]),axis=1)
    #     elif self.name == 'compass':
    #         enc_jce_data_df = pd.concat((enc_bin_data[self.bin_enc_cols[:2]],num_data[self.numerical[0]],
    #                             enc_bin_data[self.bin_enc_cols[2:]],num_data[self.numerical[1]]),axis=1)
    #     return enc_jce_data_df

    def filter_undesired_class(self, model):
        """
        DESCRIPTION:                        Method that obtains the undesired class instances according to the selected model
        
        INPUT:
        model:                              Model object containing the trained models
        
        OUTPUT: (None)
        undesired_test_df:                  DataFrame containing the original instances with the undesired predicted label
        undesired_test_np:                  Numpy array containing the original instances with the undesired predicted label
        undesired_transformed_test_df:      DataFrame containing the original instances with the undesired predicted label
        undesired_transformed_test_np:      Numpy array containing the original instances with the undesired predicted label
        undesired_test_target:              Ground truth label of the instances predicted with the undesired label
        """
        undesired_test_df = self.test_df.copy()
        undesired_transformed_test_df = self.transformed_test_df.copy()
        undesired_test_df['pred'] = model.sel.predict(self.transformed_test_df)
        undesired_test_df = undesired_test_df.loc[undesired_test_df['pred'] == self.undesired_class]
        undesired_test_target = self.test_target.loc[undesired_test_df['pred'] == self.undesired_class]
        undesired_transformed_test_df = undesired_transformed_test_df.loc[undesired_test_df['pred'] == self.undesired_class]
        del undesired_test_df['pred']
        undesired_test_np = undesired_test_df.to_numpy()
        undesired_transformed_test_np = undesired_transformed_test_df.to_numpy()
        self.undesired_test_df = undesired_test_df
        self.undesired_test_np = undesired_test_np
        self.undesired_transformed_test_df = undesired_transformed_test_df
        self.undesired_transformed_test_np = undesired_transformed_test_np
        self.undesired_test_target = undesired_test_target
        
    def change_targets_to_numpy(self):
        """
        Method that changes the targets to numpy if they are dataframes
        """
        if isinstance(self.train_target, pd.Series) or isinstance(self.train_target, pd.DataFrame):
            self.train_target = self.train_target.to_numpy().reshape((len(self.train_target.to_numpy()),))
        if isinstance(self.test_target, pd.Series) or isinstance(self.test_target, pd.DataFrame):
            self.test_target = self.test_target.to_numpy().reshape((len(self.test_target.to_numpy()),))
        if isinstance(self.test_undesired_target, pd.Series) or isinstance(self.test_undesired_target, pd.DataFrame):
            self.test_undesired_target = self.test_undesired_target.to_numpy().reshape((len(self.test_undesired_target.to_numpy()),))

    def add_test_predictions(self,predictions):
        """
        Method to add the test data predictions from a model
        Input predictions: Predictions for the test dataset
        """
        self.test_pred = predictions
    
    def add_sorted_train_data(self,instance):
        """
        Method to add/change a sorted array of the training dataset according to distance from an instance
        Input instance: Instance of interest from which to calculate all the distances
        """
        self.train_sorted = sort_data_distance(instance,self.jce_train_np,self.train_target) 

    def undesired_class_data(self):
        """
        Method to obtain the undesired class
        """
        if self.name in ['german','credit','compass']:
            undesired_class = 1
        elif self.name in ['adult','kdd_census','dutch','bank','diabetes','student','oulad','law']:
            undesired_class = 0
        return undesired_class

    def from_carla(self, df_instance):
        """
        DESCRIPTION:        Method to transform from the CARLA instance format to the normal instance format
        
        INPUT:
        df_instance:        DataFrame instance of interest to change from the CARLA framework format to normal format
        
        OUTPUT:
        normal_instance:    Dataframe containing the instance in the normal format
        """
        if len(self.carla_categorical) > 0:
            df_categorical = pd.DataFrame(self.carla_enc.inverse_transform(df_instance[self.carla_enc_cols]),columns=self.carla_categorical)
        else:
            df_categorical = pd.DataFrame()
        df_continuous = pd.DataFrame(self.carla_scaler.inverse_transform(df_instance[self.carla_continuous]),columns=self.carla_continuous)
        df = pd.concat((df_continuous, df_categorical),axis=1)
        bin_data, cat_data, num_data = pd[self.binary], pd[self.categorical], pd[self.numerical]
        enc_bin_data, enc_cat_data = self.bin_enc.transform(bin_data).toarray(), self.cat_enc.transform(cat_data).toarray()
        enc_bin_data = pd.DataFrame(enc_bin_data,index=bin_data.index, columns=self.bin_enc_cols)
        enc_cat_data = pd.DataFrame(enc_cat_data,index=cat_data.index, columns=self.cat_enc_cols)
        scaled_num_data = pd.DataFrame(self.scaler.transform(num_data), index=num_data.index, columns=self.numerical)
        normal_instance = pd.concat((enc_bin_data, enc_cat_data, scaled_num_data),axis=1)
        return normal_instance

    def store_test_undesired(self):
        """
        Method that stores the test_data_undesired information
        """
        pickle.dump(self.jce_test_undesired_df, open(f'{dataset_dir}{self.name}/{self.name}_jce_test_undesired.pkl', 'wb'))

    def define_feat_type(self):
        """
        Method that obtains a feature type vector corresponding to each of the features
        Output feat_type: Dataset feature type series
        """
        feat_type = self.jce_train_df.dtypes
        feat_type_out = copy.deepcopy(feat_type)
        feat_list = feat_type.index.tolist()
        if self.name == 'adult':
            for i in feat_list:
                if 'Sex' in i or 'Native' in i or 'WorkClass' in i or 'Marital' in i or 'Occupation' in i or 'Relation' in i or 'Race' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'EducationLevel' in i or 'Age' in i:
                    feat_type_out.loc[i] = 'num-ord'
                elif 'EducationNumber' in i or 'Capital' in i or 'Hours' in i:
                    feat_type_out.loc[i] = 'num-con'
        elif self.name == 'kdd_census':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i or 'Industry' in i or 'Occupation' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Age' in i or 'WageHour' in i or 'CapitalGain' in i or 'CapitalLoss' in i or 'Dividends' in i or 'WorkWeeksYear' in i:
                    feat_type_out.loc[i] = 'num-con'
        elif self.name == 'german':
            for i in feat_list:
                if 'Sex' in i or 'Single' in i or 'Unemployed' in i or 'Housing' in i or 'PurposeOfLoan' in i or 'InstallmentRate' in i or 'Housing' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Age' in i or 'Credit' in i or 'Loan' in i:
                    feat_type_out.loc[i] = 'num-con'
        elif self.name == 'dutch':
            for i in feat_list:
                if 'Sex' in i or 'HouseholdPosition' in i or 'HouseholdSize' in i or 'Country' in i or 'EconomicStatus' in i or 'CurEcoActivity' in i or 'MaritalStatus' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'EducationLevel' in i:
                    feat_type_out.loc[i] = 'num-ord'
                elif 'Age' in i:
                    feat_type_out.loc[i] = 'num-con'
        elif self.name == 'bank':
            for i in feat_list:
                if 'Default' in i or 'Housing' in i or 'Loan' in i or 'Job' in i or 'MaritalStatus' in i or 'Education' in i or 'Contact' in i or 'Month' in i or 'Poutcome' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Age' in i:
                    feat_type_out.loc[i] = 'num-ord'
                elif 'Balance' in i or 'Day' in i or 'Duration' in i or 'Campaign' in i or 'Pdays' in i or 'Previous' in i:
                    feat_type_out.loc[i] = 'num-con'
        elif self.name == 'credit':
            for i in feat_list:
                if 'Male' in i or 'Married' in i or 'History' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Amount' in i or 'Balance' in i or 'Spending' in i:
                    feat_type_out.loc[i] = 'num-con'
                elif 'Total' in i or 'Age' in i or 'Education' in i:
                    feat_type_out.loc[i] = 'num-ord'
        elif self.name == 'compass':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i or 'Charge' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Priors' in i or 'Age' in i:
                    feat_type_out.loc[i] = 'num-ord'
        elif self.name == 'diabetes':
            for i in feat_list:
                if 'DiabetesMed' in i or 'Race' in i or 'Sex' in i or 'A1CResult' in i or 'Metformin' in i or 'Chlorpropamide' in i or 'Glipizide' in i or 'Rosiglitazone' in i or 'Acarbose' in i or 'Miglitol' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'AgeGroup' in i:
                    feat_type_out.loc[i] = 'num-ord'
                elif 'TimeInHospital' in i or 'NumProcedures' in i or 'NumMedications' in i or 'NumEmergency':
                    feat_type_out.loc[i] = 'num-con'
        elif self.name == 'student':
            for i in feat_list:
                if 'Age' in i or 'School' in i or 'Sex' in i or 'Address' in i or 'FamilySize' in i or 'ParentStatus' in i or 'SchoolSupport' in i or 'FamilySupport' in i or 'ExtraPaid' in i or 'ExtraActivities' in i or 'Nursery' in i or 'HigherEdu' in i or 'Internet' in i or 'Romantic' in i or 'MotherJob' in i or 'FatherJob' in i or 'SchoolReason' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'MotherEducation' in i or 'FatherEducation' in i:
                    feat_type_out.loc[i] = 'num-ord'
                elif 'TravelTime' in i or 'ClassFailures' in i or 'GoOut' in i:
                    feat_type_out.loc[i] = 'num-con'
        elif self.name == 'oulad':
            for i in feat_list:
                if 'Sex' in i or 'Disability' in i or 'Region' in i or 'CodeModule' in i or 'CodePresentation' in i or 'HighestEducation' in i or 'IMDBand' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'NumPrevAttempts' in i or 'StudiedCredits' in i:
                    feat_type_out.loc[i] = 'num-con'
                elif 'AgeGroup' in i:
                    feat_type_out.loc[i] = 'num-ord'
        elif self.name == 'law':
            for i in feat_list:        
                if 'FamilyIncome' in i or 'Tier' in i or 'Race' in i or 'WorkFullTime' in i or 'Sex' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Decile1stYear' in i or 'Decile3rdYear' in i or 'LSAT' in i or 'UndergradGPA' in i or 'FirstYearGPA' in i or 'CumulativeGPA' in i:
                    feat_type_out.loc[i] = 'num-con'
        return feat_type_out

    def define_protected(self):
        """
        Method that defines which features are sensitive / protected and the groups or categories in each of them
        Output feat_protected: Protected set of features per dataset
        """
        feat_protected_values = {}
        if self.name == 'adult':
            feat_protected_values['Sex'] = {1.00:'Male', 2.00:'Female'}
            feat_protected_values['Race'] = {1.00:'White', 2.00:'Non-white'}
            feat_protected_values['AgeGroup'] = {1.00:'<25', 2.00:'25-60', 3.00:'>60'}
        elif self.name == 'kdd_census':
            feat_protected_values['Sex'] = {1.00:'Male', 2.00:'Female'}
            feat_protected_values['Race'] = {1.00:'White', 2.00:'Non-white'}
        elif self.name == 'german':
            feat_protected_values['Sex'] = {1.00:'Male', 2.00:'Female'}
        elif self.name == 'dutch':
            feat_protected_values['Sex'] = {1.00:'Male', 2.00:'Female'}
        elif self.name == 'bank':
            feat_protected_values['AgeGroup'] = {1.00:'<25', 2.00:'25-60', 3.00:'>60'}
            feat_protected_values['MaritalStatus'] = {1.00:'Married', 2.00:'Single', 3.00:'Divorced'}
        elif self.name == 'credit':
            feat_protected_values['isMale'] = {1.00:'True', 0.00:'False'}
            feat_protected_values['isMarried'] = {1.00:'True', 0.00:'False'}
            feat_protected_values['EducationLevel'] = {1.00:'Other', 2.00:'HS', 3.00:'University', 4.00:'Graduate'}
        elif self.name == 'compass':
            feat_protected_values['Race'] = {1.00:'African-American', 2.00:'Caucasian'}
            feat_protected_values['Sex'] = {1.00:'Male', 2.00:'Female'}
        elif self.name == 'diabetes':
            feat_protected_values['Sex'] = {1.00:'Male', 2.00:'Female'}
        elif self.name == 'student':
            feat_protected_values['Sex'] = {1.00:'Male', 2.00:'Female'}
            feat_protected_values['AgeGroup'] = {1.00:'<18', 2.00:'>=18'}
        elif self.name == 'oulad':
            feat_protected_values['Sex'] = {1.00:'Male', 2.00:'Female'}
        elif self.name == 'law':
            feat_protected_values['Sex'] = {1.00:'Male', 2.00:'Female'}
            feat_protected_values['Race'] = {1.00:'White', 2.00:'Non-white'}
        return feat_protected_values

    def define_mutability(self):
        """
        Method that outputs mutable features per dataset
        Output feat_mutable: Series indicating the mutability of each feature
        """
        feat_list = self.feat_type.index.tolist()
        feat_mutable  = dict()
        for i in feat_list:
            feat_mutable[i] = 1
        for i in self.feat_protected.keys():
            idx_feat_protected = [j for j in range(len(feat_list)) if i in feat_list[j]]
            feat = [feat_list[j] for j in idx_feat_protected]
            for j in feat:
                feat_mutable[j] = 0
        feat_mutable = pd.Series(feat_mutable)
        return feat_mutable
        
    def define_directionality(self):
        """
        Method that outputs change directionality of features per dataset
        Output feat_dir: Series containing plausible direction of change of each feature
        """
        feat_list = self.feat_type.index.tolist()
        feat_dir  = dict()
        if self.name == 'adult':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'Race' in i:
                    feat_dir[i] = 0
                elif 'Education' in i:
                    feat_dir[i] = 'pos'
                else:
                    feat_dir[i] = 'any'
        elif self.name == 'kdd_census':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_dir[i] = 0
                elif 'Industry' in i or 'Occupation' in i or 'WageHour' in i or 'CapitalGain' in i or 'CapitalLoss' in i or 'Dividends' in i or 'WorkWeeksYear' or 'Age' in i:
                    feat_dir[i] = 'any'
        elif self.name == 'german':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i:
                    feat_dir[i] = 0
                else:
                    feat_dir[i] = 'any'
        elif self.name == 'dutch':
            for i in feat_list:
                if 'Sex' in i:
                    feat_dir[i] = 0
                elif 'HouseholdPosition' in i or 'HouseholdSize' in i or 'EconomicStatus' in i or 'CurEcoActivity' in i or 'MaritalStatus' in i or 'Country' in i:
                    feat_dir[i] = 'any'
                elif 'EducationLevel' in i or 'Age' in i:
                    feat_dir[i] = 'pos'
        elif self.name == 'bank':
            for i in feat_list:
                if 'Age' in i or 'Marital' in i:
                    feat_dir[i] = 0
                elif 'Default' in i or 'Housing' in i or 'Loan' in i or 'Job' in i or 'Contact' in i or 'Month' in i or 'Poutcome' or 'Balance' in i or 'Day' in i or 'Duration' in i or 'Campaign' in i or 'Pdays' in i or 'Previous' in i:
                    feat_dir[i] = 'any'
                elif 'Education' in i:
                    feat_dir[i] = 'pos'
        elif self.name == 'credit':
            for i in feat_list:
                if 'Age' in i or 'Male' in i:
                    feat_dir[i] = 0
                elif 'OverLast6Months' in i or 'MostRecent' in i or 'Total' in i or 'History' in i or 'Married' in i:
                    feat_dir[i] = 'any'   
                elif 'Education' in i:
                    feat_dir[i] = 'pos'
        elif self.name == 'compass':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'Race' in i:
                    feat_dir[i] = 0
                elif 'Charge' in i or 'Priors' in i:
                    feat_dir[i] = 'any'
        elif self.name == 'diabetes':
            for i in feat_list:
                if 'Sex' in i:
                    feat_dir[i] = 0
                else:
                    feat_dir[i] = 'any'
        elif self.name == 'student':
            for i in feat_list:
                if 'Sex' in i or 'Age' in i:
                    feat_dir[i] = 0
                else:
                    feat_dir[i] = 'any'
        elif self.name == 'oulad':
            for i in feat_list:
                if 'Sex' in i:
                    feat_dir[i] = 0
                else:
                    feat_dir[i] = 'any'
        elif self.name == 'law':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_dir[i] = 0
                else:
                    feat_dir[i] = 'any'
        feat_dir = pd.Series(feat_dir)
        return feat_dir

    def define_feat_cost(self):
        """
        Method that allocates a unit cost of change to the features of the datasets
        Output feat_cost: Series with the theoretical unit cost of changing each feature
        """
        feat_cost  = dict()
        feat_list = self.feat_type.index.tolist()
        if self.name == 'adult':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'Native' in i or 'Race' in i:
                    feat_cost[i] = 0
                elif 'EducationLevel' in i:
                    feat_cost[i] = 1#50
                elif 'EducationNumber' in i:
                    feat_cost[i] = 1#20
                elif 'WorkClass' in i:
                    feat_cost[i] = 1#10
                elif 'Capital' in i:
                    feat_cost[i] = 1#5
                elif 'Hours' in i:
                    feat_cost[i] = 1#2
                elif 'Marital' in i:
                    feat_cost[i] = 1#50
                elif 'Occupation' in i:
                    feat_cost[i] = 1#10
                elif 'Relationship' in i:
                    feat_cost[i] = 1#50
        elif self.name == 'kdd_census':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_cost[i] = 0
                elif 'Industry' in i or 'Occupation' in i or 'WageHour' in i or 'CapitalGain' in i or 'CapitalLoss' in i or 'Dividends' in i or 'WorkWeeksYear':
                    feat_cost[i] = 1
        elif self.name == 'german':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i:
                    feat_cost[i] = 0
                else:
                    feat_cost[i] = 1
        elif self.name == 'dutch':
            for i in feat_list:
                if 'Sex' in i:
                    feat_cost[i] = 0
                elif 'HouseholdPosition' in i or 'HouseholdSize' in i or 'EconomicStatus' in i or 'CurEcoActivity' in i or 'MaritalStatus' in i or 'EducationLevel' in i or 'Age' in i or 'Country' in i:
                    feat_cost[i] = 1
        elif self.name == 'bank':
            for i in feat_list:
                if 'Age' in i or 'Marital' in i:
                    feat_cost[i] = 0
                else:
                    feat_cost[i] = 1
        elif self.name == 'credit':
            for i in feat_list:
                if 'Age' in i or 'Male' in i:
                    feat_cost[i] = 0
                elif 'OverLast6Months' in i or 'MostRecent' in i or 'TotalOverdueCounts' in i or 'History' in i:
                    feat_cost[i] = 1#20
                elif 'TotalMonthsOverdue' in i:
                    feat_cost[i] = 1#10   
                elif 'Education' in i:
                    feat_cost[i] = 1#50
                elif 'Married' in i:
                    feat_cost[i] = 1#50
        elif self.name == 'compass':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'Race' in i:
                    feat_cost[i] = 0
                elif 'Charge' in i:
                    feat_cost[i] = 1#10
                elif 'Priors' in i:
                    feat_cost[i] = 1#20
        elif self.name == 'diabetes':
            for i in feat_list:
                if 'Sex' in i:
                    feat_cost[i] = 0
                else:
                    feat_cost[i] = 1
        elif self.name == 'student':
            for i in feat_list:
                if 'Sex' in i or 'Age' in i:
                    feat_cost[i] = 0
                else:
                    feat_cost[i] = 1
        elif self.name == 'oulad':
            for i in feat_list:
                if 'Sex' in i:
                    feat_cost[i] = 0
                else:
                    feat_cost[i] = 1
        elif self.name == 'law':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_cost[i] = 0
                else:
                    feat_cost[i] = 1
        feat_cost = pd.Series(feat_cost)
        return feat_cost

    def define_feat_step(self):
        """
        Method that estimates the step size of all features (used for ordinal features)
        Output feat_step: Plausible step size for each feature 
        """
        feat_step = pd.Series(data=1/(self.jce_scaler.data_max_ - self.jce_scaler.data_min_),index=[i for i in self.feat_type.keys() if self.feat_type[i] in ['num-ord','num-con']])
        for i in self.feat_type.keys().tolist():
            if self.feat_type.loc[i] == 'num-con':
                feat_step.loc[i] = self.step
            elif self.feat_type.loc[i] == 'num-ord':
                continue
            else:
                feat_step.loc[i] = 0
        feat_step = feat_step.reindex(index = self.feat_type.keys().to_list())
        return feat_step

    def define_category_groups(self):
        """
        Method that assigns categorical groups to different one-hot encoded categorical features
        Output feat_cat: Category groups for each of the features
        """
        feat_cat = copy.deepcopy(self.feat_type)
        feat_list = self.feat_type.index.tolist()
        if self.name == 'adult':
            for i in feat_list:
                if 'Sex' in i or 'Native' in i or 'EducationLevel' or i in 'EducationNumber' in i or 'Capital' in i or 'Hours' in i or 'Race' in i:
                    feat_cat.loc[i] = 'non'
                elif 'Age' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'WorkClass' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Marital' in i:
                    feat_cat.loc[i] = 'cat_2'
                elif 'Occupation' in i:
                    feat_cat.loc[i] = 'cat_3'
                elif 'Relation' in i:
                    feat_cat.loc[i] = 'cat_4'
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'kdd_census':
            for i in feat_list:
                if 'Industry' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'Occupation' in i:
                    feat_cat.loc[i] = 'cat_1'    
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'german':
            for i in feat_list:
                if 'PurposeOfLoan' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'InstallmentRate' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Housing' in i:
                    feat_cat.loc[i] = 'cat_2'
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'dutch':
            for i in feat_list:
                if 'HouseholdPosition' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'HouseholdSize' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Country' in i:
                    feat_cat.loc[i] = 'cat_2'
                elif 'EconomicStatus' in i:
                    feat_cat.loc[i] = 'cat_3'
                elif 'CurEcoActivity' in i:
                    feat_cat.loc[i] = 'cat_4'
                elif 'MaritalStatus' in i:
                    feat_cat.loc[i] = 'cat_5'
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'bank':
            for i in feat_list:
                if 'Job' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'MaritalStatus' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Education' in i:
                    feat_cat.loc[i] = 'cat_2'
                elif 'Contact' in i:
                    feat_cat.loc[i] = 'cat_3'
                elif 'Month' in i:
                    feat_cat.loc[i] = 'cat_4'
                elif 'Poutcome' in i:
                    feat_cat.loc[i] = 'cat_5'
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'credit':
            for i in feat_list:
                feat_cat.loc[i] = 'non'
        elif self.name == 'compass':
            for i in feat_list:
                feat_cat.loc[i] = 'non'
        elif self.name == 'diabetes':
            for i in feat_list:
                if 'Race' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'A1CResult' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Metformin' in i:
                    feat_cat.loc[i] = 'cat_2'
                elif 'Chlorpropamide' in i:
                    feat_cat.loc[i] = 'cat_3'
                elif 'Glipizide' in i:
                    feat_cat.loc[i] = 'cat_4'
                elif 'Rosiglitazone' in i:
                    feat_cat.loc[i] = 'cat_5'
                elif 'Acarbose' in i:
                    feat_cat.loc[i] = 'cat_6'
                elif 'Miglitol' in i:
                    feat_cat.loc[i] = 'cat_7'
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'student':
            for i in feat_list:
                if 'MotherJob' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'FatherJob' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'SchoolReason' in i:
                    feat_cat.loc[i] = 'cat_2'
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'oulad':
            for i in feat_list:
                if 'Region' in i:
                    feat_cat.loc[i] = 'cat_0'
                elif 'CodeModule' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'CodePresentation' in i:
                    feat_cat.loc[i] = 'cat_2'
                elif 'HighestEducation' in i:
                    feat_cat.loc[i] = 'cat_3'
                elif 'IMDBand' in i:
                    feat_cat.loc[i] = 'cat_4'
                else:
                    feat_cat.loc[i] = 'non'    
        elif self.name == 'law':
            for i in feat_list:
                if 'FamilyIncome' in i:   
                    feat_cat.loc[i] = 'cat_0'
                elif 'Tier' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Race' in i:
                    feat_cat.loc[i] = 'cat_2'
                else:
                    feat_cat.loc[i] = 'non'    
        return feat_cat

class Model:
    """
    Class that contains the trained models
    """
    def __init__(self,data_obj,grid_search_path):
        self.model_params_path = grid_search_path
        self.train_clf_model(data_obj)
    
    def train_clf_model(self,data_obj):
        """
        Method that trains the classifier model according to the data object received
        """
        grid_search_results = pd.read_csv(str(self.model_params_path)+'/Results/grid_search/grid_search_final.csv',index_col = ['dataset','model'])
        sel_model_str, params_best, params_rf = best_model_params(grid_search_results,data_obj.name)
        self.jce_sel, self.jce_rf = clf_model(sel_model_str,params_best,params_rf,data_obj.jce_train_df,data_obj.train_target)
        self.carla_sel, self.carla_rf = clf_model(sel_model_str,params_best,params_rf,data_obj.carla_train_df,data_obj.train_target)

def erase_missing(data):
    """
    Function that eliminates instances with missing values
    Input data: The dataset of interest
    Output data: Filtered dataset without points with missing values
    """
    data = data.replace({'?':np.nan})
    data = data.replace({' ?':np.nan})
    data.dropna(axis=0,how='any',inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def verify_column(data):
    """
    Function that verifies whether a column has equal values. If yes, eliminates it from the dataset
    Input data: Dataset analyzed.
    Output data: Dataset modified if a column is found to have all values equal.
    """
    for i in data.columns:
        if len(data[i].unique().tolist()) == 1:
            del data[i]
    return data  

def eliminate_columns(data):
    """
    Function to eliminate to-be-erased columns from the datasets
    Input data: The training dataset to encode the categorical features (has to be in Pandas dataframe)
    Output data: Dataset with eliminated columns
    """
    for i in data.columns:
        if i.find('to be erased') != -1 or i.find('to be deleted') != -1 or len(data[i].unique()) == 1:
            data.drop(columns=i,inplace=True)
    return data

def erase_duplicates(data):
    """
    Function that eliminates duplicate instances
    Input data: The dataset of interest.
    Output data: Filtered dataset without duplicate instances
    """
    label = data['class']
    data.drop(columns=['class'],inplace=True)
    data.drop_duplicates(inplace = True)
    data['class'] = label
    data.reset_index(inplace = True)
    data.drop(columns=['index'],inplace=True)
    return data

def nom_to_num(data):
    """
    Function to transform categorical features into encoded numerical values.
    Input data: The dataset to encode the categorical features.
    Output data: The dataset with categorical features encoded into numerical features.
    """
    encoder = LabelEncoder()
    if data['label'].dtypes == object or data['label'].dtypes == str:
        encoder.fit(data['label'])
        data['label'] = encoder.transform(data['label'])
    return data, encoder

def load_model_dataset(data_str,train_fraction,seed,step,path_here = None):
    """
    Function to load all datasets according to data_str and train_fraction, and the corresponding selected and RF models for counterfactual search
    Input data_str: Name of the dataset to load
    Input train_fraction: Percentage of dataset instances to use as training dataset
    Input seed: Random seed to be used
    Input step: Size of the step to be used for continuous variable changes
    Input path: Path to the grid search results for model parameter selection
    Output data_obj: Dataset object
    Output model_obj: Model object
    """
    if data_str == 'adult':
        # Based on the MACE algorithm Preprocessing (please, see: https://github.com/amirhk/mace)
        binary = ['Sex','NativeCountry','Race']
        categorical = ['WorkClass','MaritalStatus','Occupation','Relationship']
        numerical = ['EducationNumber','CapitalGain','CapitalLoss','HoursPerWeek','EducationLevel','AgeGroup']
        label = ['label']
        carla_categorical = ['Sex','AgeGroup','Race','NativeCountry','WorkClass','MaritalStatus','Occupation','Relationship']
        carla_continuous = ['EducationNumber','CapitalGain','CapitalLoss','HoursPerWeek','EducationLevel']
        attrs = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']  # all attributes
        sensitive_attrs = ['sex']  # the fairness constraints will be used for this feature
        attrs_to_ignore = ['sex','fnlwgt']  #
        this_files_directory = dataset_dir+data_str+'/'
        data_files = ["adult.data", "adult.test"]
        X = []
        y = []
        x_control = {}
        attrs_to_vals = {}
        for k in attrs:
            if k in sensitive_attrs:
                x_control[k] = []
            elif k in attrs_to_ignore:
                pass
            else:
                attrs_to_vals[k] = []
        for file_name in data_files:
            full_file_name = os.path.join(this_files_directory, file_name)
            for line in open(full_file_name):
                line = line.strip()
                if line == "":
                    continue
                line = line.split(", ")
                if len(line) != 15 or "?" in line:
                    continue
                class_label = line[-1]
                if class_label in ["<=50K.", "<=50K"]:
                    class_label = 0
                elif class_label in [">50K.", ">50K"]:
                    class_label = +1
                else:
                    raise Exception("Invalid class label value")
                y.append(class_label)
                for i in range(0, len(line) - 1):
                    attr_name = attrs[i]
                    attr_val = line[i]
                    if attr_name == "native_country":
                        if attr_val != "United-States":
                            attr_val = "Non-United-Stated"
                    elif attr_name == "education":
                        if attr_val in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
                            attr_val = "prim-middle-school"
                        elif attr_val in ["9th", "10th", "11th", "12th"]:
                            attr_val = "high-school"
                    elif attr_name == 'race':
                        if attr_val != 'White':
                            attr_val = 'Non-white'
                    elif attr_name == 'age':
                        if int(attr_val) < 25:
                            attr_val = 1
                        elif int(attr_val) >= 25 and int(attr_val) <= 60:
                            attr_val = 2
                        elif int(attr_val) > 60:
                            attr_val = 3
                    if attr_name in sensitive_attrs:
                        x_control[attr_name].append(attr_val)
                    elif attr_name in attrs_to_ignore:
                        pass
                    else:
                        attrs_to_vals[attr_name].append(attr_val)
        all_attrs_to_vals = attrs_to_vals
        all_attrs_to_vals['sex'] = x_control['sex']
        all_attrs_to_vals['label'] = y
        first_key = list(all_attrs_to_vals.keys())[0]
        for key in all_attrs_to_vals.keys():
            assert (len(all_attrs_to_vals[key]) == len(all_attrs_to_vals[first_key]))
        df = pd.DataFrame.from_dict(all_attrs_to_vals)
        processed_df = pd.DataFrame()
        processed_df['label'] = df['label']
        processed_df.loc[df['sex'] == 'Male', 'Sex'] = 1
        processed_df.loc[df['sex'] == 'Female', 'Sex'] = 2
        processed_df.loc[df['race'] == 'White', 'Race'] = 1
        processed_df.loc[df['race'] == 'Non-white', 'Race'] = 2
        processed_df.loc[df['age'] == 1, 'AgeGroup'] = 1
        processed_df.loc[df['age'] == 2, 'AgeGroup'] = 2
        processed_df.loc[df['age'] == 3, 'AgeGroup'] = 3
        processed_df.loc[df['native_country'] == 'United-States', 'NativeCountry'] = 1
        processed_df.loc[df['native_country'] == 'Non-United-Stated', 'NativeCountry'] = 2
        processed_df.loc[df['workclass'] == 'Federal-gov', 'WorkClass'] = 1
        processed_df.loc[df['workclass'] == 'Local-gov', 'WorkClass'] = 2
        processed_df.loc[df['workclass'] == 'Private', 'WorkClass'] = 3
        processed_df.loc[df['workclass'] == 'Self-emp-inc', 'WorkClass'] = 4
        processed_df.loc[df['workclass'] == 'Self-emp-not-inc', 'WorkClass'] = 5
        processed_df.loc[df['workclass'] == 'State-gov', 'WorkClass'] = 6
        processed_df.loc[df['workclass'] == 'Without-pay', 'WorkClass'] = 7
        processed_df['EducationNumber'] = df['education_num'].astype(int)
        processed_df.loc[df['education'] == 'prim-middle-school', 'EducationLevel'] = int(1)
        processed_df.loc[df['education'] == 'high-school', 'EducationLevel'] = int(2)
        processed_df.loc[df['education'] == 'HS-grad', 'EducationLevel'] = int(3)
        processed_df.loc[df['education'] == 'Some-college', 'EducationLevel'] = int(4)
        processed_df.loc[df['education'] == 'Bachelors', 'EducationLevel'] = int(5)
        processed_df.loc[df['education'] == 'Masters', 'EducationLevel'] = int(6)
        processed_df.loc[df['education'] == 'Doctorate', 'EducationLevel'] = int(7)
        processed_df.loc[df['education'] == 'Assoc-voc', 'EducationLevel'] = int(8)
        processed_df.loc[df['education'] == 'Assoc-acdm', 'EducationLevel'] = int(9)
        processed_df.loc[df['education'] == 'Prof-school', 'EducationLevel'] = int(10)
        processed_df.loc[df['marital_status'] == 'Divorced', 'MaritalStatus'] = 1
        processed_df.loc[df['marital_status'] == 'Married-AF-spouse', 'MaritalStatus'] = 2
        processed_df.loc[df['marital_status'] == 'Married-civ-spouse', 'MaritalStatus'] = 3
        processed_df.loc[df['marital_status'] == 'Married-spouse-absent', 'MaritalStatus'] = 4
        processed_df.loc[df['marital_status'] == 'Never-married', 'MaritalStatus'] = 5
        processed_df.loc[df['marital_status'] == 'Separated', 'MaritalStatus'] = 6
        processed_df.loc[df['marital_status'] == 'Widowed', 'MaritalStatus'] = 7
        processed_df.loc[df['occupation'] == 'Adm-clerical', 'Occupation'] = 1
        processed_df.loc[df['occupation'] == 'Armed-Forces', 'Occupation'] = 2
        processed_df.loc[df['occupation'] == 'Craft-repair', 'Occupation'] = 3
        processed_df.loc[df['occupation'] == 'Exec-managerial', 'Occupation'] = 4
        processed_df.loc[df['occupation'] == 'Farming-fishing', 'Occupation'] = 5
        processed_df.loc[df['occupation'] == 'Handlers-cleaners', 'Occupation'] = 6
        processed_df.loc[df['occupation'] == 'Machine-op-inspct', 'Occupation'] = 7
        processed_df.loc[df['occupation'] == 'Other-service', 'Occupation'] = 8
        processed_df.loc[df['occupation'] == 'Priv-house-serv', 'Occupation'] = 9
        processed_df.loc[df['occupation'] == 'Prof-specialty', 'Occupation'] = 10
        processed_df.loc[df['occupation'] == 'Protective-serv', 'Occupation'] = 11
        processed_df.loc[df['occupation'] == 'Sales', 'Occupation'] = 12
        processed_df.loc[df['occupation'] == 'Tech-support', 'Occupation'] = 13
        processed_df.loc[df['occupation'] == 'Transport-moving', 'Occupation'] = 14
        processed_df.loc[df['relationship'] == 'Husband', 'Relationship'] = 1
        processed_df.loc[df['relationship'] == 'Not-in-family', 'Relationship'] = 2
        processed_df.loc[df['relationship'] == 'Other-relative', 'Relationship'] = 3
        processed_df.loc[df['relationship'] == 'Own-child', 'Relationship'] = 4
        processed_df.loc[df['relationship'] == 'Unmarried', 'Relationship'] = 5
        processed_df.loc[df['relationship'] == 'Wife', 'Relationship'] = 6
        processed_df['CapitalGain'] = df['capital_gain'].astype(int)
        processed_df['CapitalLoss'] = df['capital_loss'].astype(int)
        processed_df['HoursPerWeek'] = df['hours_per_week'].astype(int)
    elif data_str == 'kdd_census':
        binary = ['Sex','Race']
        categorical = ['Industry','Occupation']
        numerical = ['Age','WageHour','CapitalGain','CapitalLoss','Dividends','WorkWeeksYear']
        label = ['Label']
        carla_categorical = ['Sex','Race','Industry','Occupation']
        carla_continuous = ['Age','WageHour','CapitalGain','CapitalLoss','Dividends','WorkWeeksYear']
        cols = binary + numerical + categorical + label
        read_cols = ['Age','WorkClass','IndustryDetail','OccupationDetail','Education','WageHour','Enrolled','MaritalStatus','Industry','Occupation',
                'Race','Hispanic','Sex','Union','UnemployedReason','FullTimePartTime','CapitalGain','CapitalLoss','Dividends','Tax',
                'RegionPrev','StatePrev','HouseDetailFamily','HouseDetailSummary','UnknownFeature','ChangeMsa','ChangeReg','MoveReg','Live1YrAgo','PrevSunbelt','NumPersonsWorkEmp',
                'Under18Family','CountryFather','CountryMother','Country','Citizenship','OwnBusiness','VeteransAdmin','VeteransBenefits','WorkWeeksYear','Year','Label']
        train_raw_df = pd.read_csv(dataset_dir+'/kdd_census/census-income.data',index_col=False,names=read_cols)
        test_raw_df = pd.read_csv(dataset_dir+'/kdd_census/census-income.test',index_col=False,names=read_cols)
        raw_df = pd.concat((train_raw_df,test_raw_df),axis=0)
        raw_df.reset_index(drop=True, inplace=True)
        processed_df = raw_df[cols]
        processed_df.loc[processed_df['Sex'] == ' Male','Sex'] = 1
        processed_df.loc[processed_df['Sex'] == ' Female','Sex'] = 2
        processed_df.loc[processed_df['Race'] != ' White','Race'] = 'Non-white'
        processed_df.loc[processed_df['Race'] == ' White','Race'] = 1
        processed_df.loc[processed_df['Race'] == 'Non-white','Race'] = 2
        processed_df.loc[processed_df['Industry'] == ' Construction','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Entertainment','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Finance insurance and real estate','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Business and repair services','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Manufacturing-nondurable goods','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Personal services except private HH','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Manufacturing-durable goods','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Other professional services','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Mining','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Transportation','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Wholesale trade','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Public administration','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Retail trade','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Social services','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Private household services','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Communications','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Agriculture','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Forestry and fisheries','Industry'] = 'Industry'
        processed_df.loc[processed_df['Industry'] == ' Education','Industry'] = 'Education'
        processed_df.loc[processed_df['Industry'] == ' Utilities and sanitary services','Industry'] = 'Medical'
        processed_df.loc[processed_df['Industry'] == ' Hospital services','Industry'] = 'Medical'
        processed_df.loc[processed_df['Industry'] == ' Medical except hospital','Industry'] = 'Medical'
        processed_df.loc[processed_df['Industry'] == ' Armed Forces','Industry'] = 'Military'
        processed_df.loc[processed_df['Industry'] == ' Not in universe or children','Industry'] = 'Other'
        processed_df.loc[processed_df['Industry'] == 'Industry','Industry'] = 1
        processed_df.loc[processed_df['Industry'] == 'Education','Industry'] = 2
        processed_df.loc[processed_df['Industry'] == 'Medical','Industry'] = 3
        processed_df.loc[processed_df['Industry'] == 'Military','Industry'] = 4
        processed_df.loc[processed_df['Industry'] == 'Other','Industry'] = 5
        processed_df.loc[processed_df['Occupation'] == ' Precision production craft & repair','Occupation'] = 'Technician'
        processed_df.loc[processed_df['Occupation'] == ' Professional specialty','Occupation'] = 'Executive'
        processed_df.loc[processed_df['Occupation'] == ' Executive admin and managerial','Occupation'] = 'Executive'
        processed_df.loc[processed_df['Occupation'] == ' Handlers equip cleaners etc ','Occupation'] = 'Services'
        processed_df.loc[processed_df['Occupation'] == ' Adm support including clerical','Occupation'] = 'Services'
        processed_df.loc[processed_df['Occupation'] == ' Machine operators assmblrs & inspctrs','Occupation'] = 'Technician'
        processed_df.loc[processed_df['Occupation'] == ' Sales','Occupation'] = 'Executive'
        processed_df.loc[processed_df['Occupation'] == ' Private household services','Occupation'] = 'Services'
        processed_df.loc[processed_df['Occupation'] == ' Technicians and related support','Occupation'] = 'Technician'
        processed_df.loc[processed_df['Occupation'] == ' Transportation and material moving','Occupation'] = 'Services'
        processed_df.loc[processed_df['Occupation'] == ' Farming forestry and fishing','Occupation'] = 'Technician'
        processed_df.loc[processed_df['Occupation'] == ' Protective services','Occupation'] = 'Services'
        processed_df.loc[processed_df['Occupation'] == ' Other service','Occupation'] = 'Services'
        processed_df.loc[processed_df['Occupation'] == ' Armed Forces','Occupation'] = 'Military'
        processed_df.loc[processed_df['Occupation'] == ' Not in universe','Occupation'] = 'Other'
        processed_df.loc[processed_df['Occupation'] == 'Technician','Occupation'] = 1
        processed_df.loc[processed_df['Occupation'] == 'Executive','Occupation'] = 2
        processed_df.loc[processed_df['Occupation'] == 'Services','Occupation'] = 3
        processed_df.loc[processed_df['Occupation'] == 'Military','Occupation'] = 4
        processed_df.loc[processed_df['Occupation'] == 'Other','Occupation'] = 5
        processed_df.loc[processed_df['Label'] == ' - 50000.','Label'] = int(0)
        processed_df.loc[processed_df['Label'] == ' 50000+.','Label'] = int(1)
        processed_df['Label']=processed_df['Label'].astype(int)
    elif data_str == 'german':
        binary = ['Sex','Single','Unemployed']
        categorical = ['PurposeOfLoan','InstallmentRate','Housing']
        numerical = ['Age','Credit','LoanDuration']
        label = ['Label']
        carla_categorical = ['Sex','Single','Unemployed','PurposeOfLoan','InstallmentRate','Housing']
        carla_continuous = ['Age','Credit','LoanDuration']
        cols = binary + numerical + categorical + label
        processed_df = pd.DataFrame()
        raw_df = pd.read_csv(dataset_dir+'/german/german_raw.csv')
        processed_df['GoodCustomer'] = raw_df['GoodCustomer']
        processed_df['PurposeOfLoan'] = raw_df['PurposeOfLoan']
        processed_df['PurposeOfLoan'] = raw_df['PurposeOfLoan']
        processed_df['Single'] = raw_df['Single']
        processed_df['Unemployed'] = raw_df['Unemployed']
        processed_df['InstallmentRate'] = raw_df['LoanRateAsPercentOfIncome']
        processed_df.loc[processed_df['GoodCustomer'] == -1,'Label'] = int(0)
        processed_df.loc[processed_df['GoodCustomer'] == 1,'Label'] = int(1)
        processed_df.loc[processed_df['PurposeOfLoan'] == 'Business','PurposeOfLoan'] = 1
        processed_df.loc[processed_df['PurposeOfLoan'] == 'Education','PurposeOfLoan'] = 2
        processed_df.loc[processed_df['PurposeOfLoan'] == 'Electronics','PurposeOfLoan'] = 3
        processed_df.loc[processed_df['PurposeOfLoan'] == 'Furniture','PurposeOfLoan'] = 4
        processed_df.loc[processed_df['PurposeOfLoan'] == 'HomeAppliances','PurposeOfLoan'] = 5
        processed_df.loc[processed_df['PurposeOfLoan'] == 'NewCar','PurposeOfLoan'] = 6
        processed_df.loc[processed_df['PurposeOfLoan'] == 'Other','PurposeOfLoan'] = 7
        processed_df.loc[processed_df['PurposeOfLoan'] == 'Repairs','PurposeOfLoan'] = 8
        processed_df.loc[processed_df['PurposeOfLoan'] == 'Retraining','PurposeOfLoan'] = 9
        processed_df.loc[processed_df['PurposeOfLoan'] == 'UsedCar','PurposeOfLoan'] = 10
        processed_df.loc[raw_df['Gender'] == 'Male', 'Sex'] = 1
        processed_df.loc[raw_df['Gender'] == 'Female', 'Sex'] = 2
        processed_df.loc[raw_df['OwnsHouse'] == 1, 'Housing'] = 1
        processed_df.loc[raw_df['RentsHouse'] == 1, 'Housing'] = 2
        processed_df.loc[(raw_df['OwnsHouse'] == 0) & (raw_df['RentsHouse'] == 0), 'Housing'] = 3
        processed_df['Age'] = raw_df['Age']
        processed_df['Credit'] = raw_df['Credit']
        processed_df['LoanDuration'] = raw_df['LoanDuration']
        processed_df['Label']=processed_df['Label'].astype('int')
        processed_df = processed_df[cols]
    elif data_str == 'dutch':
        binary = ['Sex']
        categorical = ['HouseholdPosition','HouseholdSize','Country','EconomicStatus','CurEcoActivity','MaritalStatus']
        numerical = ['Age','EducationLevel']
        label = ['Occupation']
        carla_categorical = ['Sex','HouseholdPosition','HouseholdSize','Country','EconomicStatus','CurEcoActivity','MaritalStatus','EducationLevel']
        carla_continuous = ['Age']
        cols = binary + numerical + categorical + label
        raw_df = pd.read_csv(dataset_dir+'/dutch/dutch.txt')
        processed_df = raw_df[cols]
        processed_df.loc[processed_df['HouseholdPosition'] == 1131,'HouseholdPosition'] = 1
        processed_df.loc[processed_df['HouseholdPosition'] == 1122,'HouseholdPosition'] = 2
        processed_df.loc[processed_df['HouseholdPosition'] == 1121,'HouseholdPosition'] = 3
        processed_df.loc[processed_df['HouseholdPosition'] == 1110,'HouseholdPosition'] = 4
        processed_df.loc[processed_df['HouseholdPosition'] == 1210,'HouseholdPosition'] = 5
        processed_df.loc[processed_df['HouseholdPosition'] == 1132,'HouseholdPosition'] = 6
        processed_df.loc[processed_df['HouseholdPosition'] == 1140,'HouseholdPosition'] = 7
        processed_df.loc[processed_df['HouseholdPosition'] == 1220,'HouseholdPosition'] = 8
        processed_df.loc[processed_df['HouseholdSize'] == 111,'HouseholdSize'] = 1
        processed_df.loc[processed_df['HouseholdSize'] == 112,'HouseholdSize'] = 2
        processed_df.loc[processed_df['HouseholdSize'] == 113,'HouseholdSize'] = 3
        processed_df.loc[processed_df['HouseholdSize'] == 114,'HouseholdSize'] = 4
        processed_df.loc[processed_df['HouseholdSize'] == 125,'HouseholdSize'] = 5
        processed_df.loc[processed_df['HouseholdSize'] == 126,'HouseholdSize'] = 6
        processed_df.loc[processed_df['EconomicStatus'] == 111,'EconomicStatus'] = 1
        processed_df.loc[processed_df['EconomicStatus'] == 120,'EconomicStatus'] = 2
        processed_df.loc[processed_df['EconomicStatus'] == 112,'EconomicStatus'] = 3
        processed_df.loc[processed_df['CurEcoActivity'] == 131,'CurEcoActivity'] = 1
        processed_df.loc[processed_df['CurEcoActivity'] == 135,'CurEcoActivity'] = 2
        processed_df.loc[processed_df['CurEcoActivity'] == 138,'CurEcoActivity'] = 3
        processed_df.loc[processed_df['CurEcoActivity'] == 122,'CurEcoActivity'] = 4
        processed_df.loc[processed_df['CurEcoActivity'] == 137,'CurEcoActivity'] = 5
        processed_df.loc[processed_df['CurEcoActivity'] == 136,'CurEcoActivity'] = 6
        processed_df.loc[processed_df['CurEcoActivity'] == 133,'CurEcoActivity'] = 7
        processed_df.loc[processed_df['CurEcoActivity'] == 139,'CurEcoActivity'] = 8
        processed_df.loc[processed_df['CurEcoActivity'] == 132,'CurEcoActivity'] = 9
        processed_df.loc[processed_df['CurEcoActivity'] == 134,'CurEcoActivity'] = 10
        processed_df.loc[processed_df['CurEcoActivity'] == 111,'CurEcoActivity'] = 11
        processed_df.loc[processed_df['CurEcoActivity'] == 124,'CurEcoActivity'] = 12
        processed_df.loc[processed_df['Occupation'] == '5_4_9','Occupation'] = int(1)
        processed_df.loc[processed_df['Occupation'] == '2_1','Occupation'] = int(0)
        processed_df['Occupation']=processed_df['Occupation'].astype('int')
        processed_df.loc[processed_df['Age'] == 4,'Age'] = 15
        processed_df.loc[processed_df['Age'] == 5,'Age'] = 16
        processed_df.loc[processed_df['Age'] == 6,'Age'] = 18
        processed_df.loc[processed_df['Age'] == 7,'Age'] = 21
        processed_df.loc[processed_df['Age'] == 8,'Age'] = 22
        processed_df.loc[processed_df['Age'] == 9,'Age'] = 27
        processed_df.loc[processed_df['Age'] == 10,'Age'] = 32
        processed_df.loc[processed_df['Age'] == 11,'Age'] = 37
        processed_df.loc[processed_df['Age'] == 12,'Age'] = 42
        processed_df.loc[processed_df['Age'] == 13,'Age'] = 47
        processed_df.loc[processed_df['Age'] == 14,'Age'] = 52
        processed_df.loc[processed_df['Age'] == 15,'Age'] = 59
    elif data_str == 'bank':
        binary = ['Default','Housing','Loan']
        categorical = ['Job','MaritalStatus','Education','Contact','Month','Poutcome']
        numerical = ['AgeGroup','Balance','Day','Duration','Campaign','Pdays','Previous']
        label = ['Subscribed']
        carla_categorical = ['Default','Housing','Loan','Job','MaritalStatus','Education','Contact','Month','Poutcome','AgeGroup']
        carla_continuous = ['Balance','Day','Duration','Campaign','Pdays','Previous']
        cols = binary + numerical + categorical + label
        processed_df = pd.read_csv(dataset_dir+'bank/bank.csv',sep=';',index_col=False)
        processed_df.loc[processed_df['age'] < 25,'AgeGroup'] = 1
        processed_df.loc[(processed_df['age'] <= 60) & (processed_df['age'] >= 25),'AgeGroup'] = 2
        processed_df.loc[processed_df['age'] > 60,'AgeGroup'] = 3
        processed_df.loc[processed_df['default'] == 'no','Default'] = 1
        processed_df.loc[processed_df['default'] == 'yes','Default'] = 2
        processed_df.loc[processed_df['housing'] == 'no','Housing'] = 1
        processed_df.loc[processed_df['housing'] == 'yes','Housing'] = 2
        processed_df.loc[processed_df['loan'] == 'no','Loan'] = 1
        processed_df.loc[processed_df['loan'] == 'yes','Loan'] = 2
        processed_df.loc[processed_df['job'] == 'management','Job'] = 1
        processed_df.loc[processed_df['job'] == 'technician','Job'] = 2
        processed_df.loc[processed_df['job'] == 'entrepreneur','Job'] = 3
        processed_df.loc[processed_df['job'] == 'blue-collar','Job'] = 4
        processed_df.loc[processed_df['job'] == 'retired','Job'] = 5
        processed_df.loc[processed_df['job'] == 'admin.','Job'] = 6
        processed_df.loc[processed_df['job'] == 'services','Job'] = 7
        processed_df.loc[processed_df['job'] == 'self-employed','Job'] = 8
        processed_df.loc[processed_df['job'] == 'unemployed','Job'] = 9
        processed_df.loc[processed_df['job'] == 'housemaid','Job'] = 10
        processed_df.loc[processed_df['job'] == 'student','Job'] = 11
        processed_df.loc[processed_df['job'] == 'unknown','Job'] = 12
        processed_df.loc[processed_df['marital'] == 'married','MaritalStatus'] = 1
        processed_df.loc[processed_df['marital'] == 'single','MaritalStatus'] = 2
        processed_df.loc[processed_df['marital'] == 'divorced','MaritalStatus'] = 3
        processed_df.loc[processed_df['education'] == 'primary','Education'] = 1
        processed_df.loc[processed_df['education'] == 'secondary','Education'] = 2
        processed_df.loc[processed_df['education'] == 'tertiary','Education'] = 3
        processed_df.loc[processed_df['education'] == 'unknown','Education'] = 4
        processed_df.loc[processed_df['contact'] == 'telephone','Contact'] = 1
        processed_df.loc[processed_df['contact'] == 'cellular','Contact'] = 2
        processed_df.loc[processed_df['contact'] == 'unknown','Contact'] = 3
        processed_df.loc[processed_df['month'] == 'jan','Month'] = 1
        processed_df.loc[processed_df['month'] == 'feb','Month'] = 2
        processed_df.loc[processed_df['month'] == 'mar','Month'] = 3
        processed_df.loc[processed_df['month'] == 'apr','Month'] = 4
        processed_df.loc[processed_df['month'] == 'may','Month'] = 5
        processed_df.loc[processed_df['month'] == 'jun','Month'] = 6
        processed_df.loc[processed_df['month'] == 'jul','Month'] = 7
        processed_df.loc[processed_df['month'] == 'ago','Month'] = 8
        processed_df.loc[processed_df['month'] == 'sep','Month'] = 9
        processed_df.loc[processed_df['month'] == 'oct','Month'] = 10
        processed_df.loc[processed_df['month'] == 'nov','Month'] = 11
        processed_df.loc[processed_df['month'] == 'dec','Month'] = 12
        processed_df.loc[processed_df['month'] == 'ago','Month'] = 8
        processed_df.loc[processed_df['poutcome'] == 'success','Poutcome'] = 1
        processed_df.loc[processed_df['poutcome'] == 'failure','Poutcome'] = 2
        processed_df.loc[processed_df['poutcome'] == 'other','Poutcome'] = 3
        processed_df.loc[processed_df['poutcome'] == 'unknown','Poutcome'] = 4
        processed_df.loc[processed_df['y'] == 'no','Subscribed'] = int(0)
        processed_df.loc[processed_df['y'] == 'yes','Subscribed'] = int(1)
        processed_df.rename({'balance':'Balance','day':'Day','duration':'Duration','campaign':'Campaign','pdays':'Pdays','previous':'Previous'}, inplace=True, axis=1)
        processed_df = processed_df[cols]
        processed_df['Subscribed']=processed_df['Subscribed'].astype('int')
    elif data_str == 'credit':
        binary = ['isMale','isMarried','HasHistoryOfOverduePayments']
        categorical = []
        numerical = ['MaxBillAmountOverLast6Months','MaxPaymentAmountOverLast6Months','MonthsWithZeroBalanceOverLast6Months',
                'MonthsWithLowSpendingOverLast6Months','MonthsWithHighSpendingOverLast6Months','MostRecentBillAmount',
                'MostRecentPaymentAmount','TotalOverdueCounts','TotalMonthsOverdue','AgeGroup','EducationLevel']
        label = ['NoDefaultNextMonth (label)']
        carla_categorical = ['isMale','isMarried','HasHistoryOfOverduePayments','AgeGroup','EducationLevel']
        carla_continuous = ['MaxBillAmountOverLast6Months','MaxPaymentAmountOverLast6Months','MonthsWithZeroBalanceOverLast6Months',
                'MonthsWithLowSpendingOverLast6Months','MonthsWithHighSpendingOverLast6Months','MostRecentBillAmount',
                'MostRecentPaymentAmount','TotalOverdueCounts','TotalMonthsOverdue']
        processed_df = pd.read_csv(dataset_dir + '/credit/credit_processed.csv') # Obtained from MACE algorithm Datasets (please, see: https://github.com/amirhk/mace)
    elif data_str == 'compass':
        # Based on the MACE algorithm Datasets preprocessing (please, see: https://github.com/amirhk/mace)
        processed_df = pd.DataFrame()
        binary = ['Race','Sex','ChargeDegree']
        categorical = []
        numerical = ['PriorsCount','AgeGroup']
        label = ['TwoYearRecid (label)']
        carla_categorical = ['Race','Sex','ChargeDegree','AgeGroup']
        carla_continuous = ['PriorsCount']
        FEATURES_CLASSIFICATION = ['age_cat','race','sex','priors_count','c_charge_degree']
        CLASS_FEATURE = 'two_year_recid'
        df = pd.read_csv(dataset_dir+'/compass/compas-scores-two-years.csv')
        df = df.dropna(subset=["days_b_screening_arrest"])
        # Data filtering and preparation (As observed in MACE algorithm and based on Propublica methodology. Please, see: https://github.com/amirhk/mace)
        tmp = \
            ((df["days_b_screening_arrest"] <= 30) & (df["days_b_screening_arrest"] >= -30)) & \
            (df["is_recid"] != -1) & \
            (df["c_charge_degree"] != "O") & \
            (df["score_text"] != "NA") & \
            ((df["race"] == "African-American") | (df["race"] == "Caucasian"))
        df = df[tmp == True]
        df = pd.concat([df[FEATURES_CLASSIFICATION],df[CLASS_FEATURE],], axis=1)
        processed_df['TwoYearRecid (label)'] = df['two_year_recid']
        processed_df.loc[df['age_cat'] == 'Less than 25', 'AgeGroup'] = 1
        processed_df.loc[df['age_cat'] == '25 - 45', 'AgeGroup'] = 2
        processed_df.loc[df['age_cat'] == 'Greater than 45', 'AgeGroup'] = 3
        processed_df.loc[df['race'] == 'African-American', 'Race'] = 1
        processed_df.loc[df['race'] == 'Caucasian', 'Race'] = 2
        processed_df.loc[df['sex'] == 'Male', 'Sex'] = 1
        processed_df.loc[df['sex'] == 'Female', 'Sex'] = 2
        processed_df['PriorsCount'] = df['priors_count']
        processed_df.loc[df['c_charge_degree'] == 'M', 'ChargeDegree'] = 1
        processed_df.loc[df['c_charge_degree'] == 'F', 'ChargeDegree'] = 2
        processed_df = processed_df.reset_index(drop=True)
    elif data_str == 'diabetes':
        binary = ['DiabetesMed']
        categorical = ['Race','Sex','A1CResult','Metformin','Chlorpropamide','Glipizide','Rosiglitazone','Acarbose','Miglitol']
        numerical = ['AgeGroup','TimeInHospital','NumProcedures','NumMedications','NumEmergency']
        label = ['Label']
        carla_categorical = ['Race','Sex','A1CResult','Metformin','Chlorpropamide','Glipizide','Rosiglitazone','Acarbose','Miglitol','DiabetesMed','AgeGroup']
        carla_continuous = ['TimeInHospital','NumProcedures','NumMedications','NumEmergency']
        raw_df = pd.read_csv(dataset_dir+'diabetes/diabetes.csv') # Requires numeric transform
        cols_to_delete = ['encounter_id','patient_nbr','weight','payer_code','medical_specialty',
                          'diag_1','diag_2','diag_3','max_glu_serum','repaglinide',
                          'nateglinide','acetohexamide','glyburide','tolbutamide','pioglitazone',
                          'troglitazone','tolazamide','examide','citoglipton','insulin',
                          'glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone',
                          'change','admission_type_id','discharge_disposition_id','admission_source_id','num_lab_procedures',
                          'number_outpatient','number_inpatient','number_diagnoses']
        raw_df.drop(cols_to_delete, inplace=True, axis=1)
        raw_df = erase_missing(raw_df)
        raw_df = raw_df[raw_df['readmitted'] != 'NO']
        processed_df = pd.DataFrame(index=raw_df.index)
        processed_df.loc[raw_df['race'] == 'Caucasian','Race'] = 1
        processed_df.loc[raw_df['race'] == 'AfricanAmerican','Race'] = 2
        processed_df.loc[raw_df['race'] == 'Hispanic','Race'] = 3
        processed_df.loc[raw_df['race'] == 'Asian','Race'] = 4
        processed_df.loc[raw_df['race'] == 'Other','Race'] = 5
        processed_df.loc[raw_df['gender'] == 'Male','Sex'] = 1
        processed_df.loc[raw_df['gender'] == 'Female','Sex'] = 2
        processed_df.loc[(raw_df['age'] == '[0-10)') | (raw_df['age'] == '[10-20)'),'AgeGroup'] = 1
        processed_df.loc[(raw_df['age'] == '[20-30)') | (raw_df['age'] == '[30-40)'),'AgeGroup'] = 2
        processed_df.loc[(raw_df['age'] == '[40-50)') | (raw_df['age'] == '[50-60)'),'AgeGroup'] = 3
        processed_df.loc[(raw_df['age'] == '[60-70)') | (raw_df['age'] == '[70-80)'),'AgeGroup'] = 4
        processed_df.loc[(raw_df['age'] == '[80-90)') | (raw_df['age'] == '[90-100)'),'AgeGroup'] = 5
        processed_df.loc[raw_df['A1Cresult'] == 'None','A1CResult'] = 1
        processed_df.loc[raw_df['A1Cresult'] == '>7','A1CResult'] = 2
        processed_df.loc[raw_df['A1Cresult'] == 'Norm','A1CResult'] = 3
        processed_df.loc[raw_df['A1Cresult'] == '>8','A1CResult'] = 4
        processed_df.loc[raw_df['metformin'] == 'No','Metformin'] = 1
        processed_df.loc[raw_df['metformin'] == 'Steady','Metformin'] = 2
        processed_df.loc[raw_df['metformin'] == 'Up','Metformin'] = 3
        processed_df.loc[raw_df['metformin'] == 'Down','Metformin'] = 4
        processed_df.loc[raw_df['chlorpropamide'] == 'No','Chlorpropamide'] = 1
        processed_df.loc[raw_df['chlorpropamide'] == 'Steady','Chlorpropamide'] = 2
        processed_df.loc[raw_df['chlorpropamide'] == 'Up','Chlorpropamide'] = 3
        processed_df.loc[raw_df['chlorpropamide'] == 'Down','Chlorpropamide'] = 4
        processed_df.loc[raw_df['glipizide'] == 'No','Glipizide'] = 1
        processed_df.loc[raw_df['glipizide'] == 'Steady','Glipizide'] = 2
        processed_df.loc[raw_df['glipizide'] == 'Up','Glipizide'] = 3
        processed_df.loc[raw_df['glipizide'] == 'Down','Glipizide'] = 4
        processed_df.loc[raw_df['rosiglitazone'] == 'No','Rosiglitazone'] = 1
        processed_df.loc[raw_df['rosiglitazone'] == 'Steady','Rosiglitazone'] = 2
        processed_df.loc[raw_df['rosiglitazone'] == 'Up','Rosiglitazone'] = 3
        processed_df.loc[raw_df['rosiglitazone'] == 'Down','Rosiglitazone'] = 4
        processed_df.loc[raw_df['acarbose'] == 'No','Acarbose'] = 1
        processed_df.loc[raw_df['acarbose'] == 'Steady','Acarbose'] = 2
        processed_df.loc[raw_df['acarbose'] == 'Up','Acarbose'] = 3
        processed_df.loc[raw_df['acarbose'] == 'Down','Acarbose'] = 4
        processed_df.loc[raw_df['miglitol'] == 'No','Miglitol'] = 1
        processed_df.loc[raw_df['miglitol'] == 'Steady','Miglitol'] = 2
        processed_df.loc[raw_df['miglitol'] == 'Up','Miglitol'] = 3
        processed_df.loc[raw_df['miglitol'] == 'Down','Miglitol'] = 4
        processed_df.loc[raw_df['diabetesMed'] == 'No','DiabetesMed'] = 0
        processed_df.loc[raw_df['diabetesMed'] == 'Yes','DiabetesMed'] = 1
        processed_df['TimeInHospital'] = raw_df['time_in_hospital']
        processed_df['NumProcedures'] = raw_df['num_procedures']
        processed_df['NumMedications'] = raw_df['num_medications']
        processed_df['NumEmergency'] = raw_df['number_emergency']
        processed_df.loc[raw_df['readmitted'] == '<30','Label'] = 0
        processed_df.loc[raw_df['readmitted'] == '>30','Label'] = 1
    elif data_str == 'student':
        binary = ['School','Sex','AgeGroup','Address','FamilySize','ParentStatus','SchoolSupport','FamilySupport','ExtraPaid','ExtraActivities','Nursery','HigherEdu','Internet','Romantic']
        categorical = ['MotherJob','FatherJob','SchoolReason']
        numerical = ['MotherEducation','FatherEducation','TravelTime','ClassFailures','GoOut']
        label = ['Grade']
        carla_categorical = binary + categorical
        carla_continuous = ['MotherEducation','FatherEducation','TravelTime','ClassFailures','GoOut']
        cols = binary + numerical + categorical + label
        raw_df = pd.read_csv(dataset_dir+'student/student.csv',sep=';')
        processed_df = pd.DataFrame(index=raw_df.index)
        processed_df.loc[raw_df['age'] < 18,'AgeGroup'] = 1
        processed_df.loc[raw_df['age'] >= 18,'AgeGroup'] = 2
        processed_df.loc[raw_df['school'] == 'GP','School'] = 1
        processed_df.loc[raw_df['school'] == 'MS','School'] = 2
        processed_df.loc[raw_df['sex'] == 'M','Sex'] = 1
        processed_df.loc[raw_df['sex'] == 'F','Sex'] = 2
        processed_df.loc[raw_df['address'] == 'U','Address'] = 1
        processed_df.loc[raw_df['address'] == 'R','address'] = 2
        processed_df.loc[raw_df['famsize'] == 'LE3','FamilySize'] = 1
        processed_df.loc[raw_df['famsize'] == 'GT3','FamilySize'] = 2
        processed_df.loc[raw_df['Pstatus'] == 'T','ParentStatus'] = 1
        processed_df.loc[raw_df['Pstatus'] == 'A','ParentStatus'] = 2
        processed_df.loc[raw_df['schoolsup'] == 'yes','SchoolSupport'] = 1
        processed_df.loc[raw_df['schoolsup'] == 'no','SchoolSupport'] = 2
        processed_df.loc[raw_df['famsup'] == 'yes','FamilySupport'] = 1
        processed_df.loc[raw_df['famsup'] == 'no','FamilySupport'] = 2
        processed_df.loc[raw_df['paid'] == 'yes','ExtraPaid'] = 1
        processed_df.loc[raw_df['paid'] == 'no','ExtraPaid'] = 2
        processed_df.loc[raw_df['activities'] == 'yes','ExtraActivities'] = 1
        processed_df.loc[raw_df['activities'] == 'no','ExtraActivities'] = 2
        processed_df.loc[raw_df['nursery'] == 'yes','Nursery'] = 1
        processed_df.loc[raw_df['nursery'] == 'no','Nursery'] = 2
        processed_df.loc[raw_df['higher'] == 'yes','HigherEdu'] = 1
        processed_df.loc[raw_df['higher'] == 'no','HigherEdu'] = 2
        processed_df.loc[raw_df['internet'] == 'yes','Internet'] = 1
        processed_df.loc[raw_df['internet'] == 'no','Internet'] = 2
        processed_df.loc[raw_df['romantic'] == 'yes','Romantic'] = 1
        processed_df.loc[raw_df['romantic'] == 'no','Romantic'] = 2
        processed_df.loc[raw_df['Medu'] == 0,'MotherEducation'] = 1
        processed_df.loc[raw_df['Medu'] == 1,'MotherEducation'] = 2
        processed_df.loc[raw_df['Medu'] == 2,'MotherEducation'] = 3
        processed_df.loc[raw_df['Medu'] == 3,'MotherEducation'] = 4
        processed_df.loc[raw_df['Medu'] == 4,'MotherEducation'] = 5
        processed_df.loc[raw_df['Fedu'] == 0,'FatherEducation'] = 1
        processed_df.loc[raw_df['Fedu'] == 1,'FatherEducation'] = 2
        processed_df.loc[raw_df['Fedu'] == 2,'FatherEducation'] = 3
        processed_df.loc[raw_df['Fedu'] == 3,'FatherEducation'] = 4
        processed_df.loc[raw_df['Fedu'] == 4,'FatherEducation'] = 5
        processed_df.loc[raw_df['Mjob'] == 'at_home','MotherJob'] = 1
        processed_df.loc[raw_df['Mjob'] == 'health','MotherJob'] = 2
        processed_df.loc[raw_df['Mjob'] == 'services','MotherJob'] = 3
        processed_df.loc[raw_df['Mjob'] == 'teacher','MotherJob'] = 4
        processed_df.loc[raw_df['Mjob'] == 'other','MotherJob'] = 5
        processed_df.loc[raw_df['Fjob'] == 'at_home','FatherJob'] = 1
        processed_df.loc[raw_df['Fjob'] == 'health','FatherJob'] = 2
        processed_df.loc[raw_df['Fjob'] == 'services','FatherJob'] = 3
        processed_df.loc[raw_df['Fjob'] == 'teacher','FatherJob'] = 4
        processed_df.loc[raw_df['Fjob'] == 'other','FatherJob'] = 5
        processed_df.loc[raw_df['reason'] == 'course','SchoolReason'] = 1
        processed_df.loc[raw_df['reason'] == 'home','SchoolReason'] = 2
        processed_df.loc[raw_df['reason'] == 'reputation','SchoolReason'] = 3
        processed_df.loc[raw_df['reason'] == 'other','SchoolReason'] = 4
        processed_df['TravelTime'] = raw_df['traveltime'].astype('int')
        processed_df['ClassFailures'] = raw_df['failures'].astype('int')
        processed_df['GoOut'] = raw_df['goout'].astype('int')
        processed_df.loc[raw_df['G3'] < 10,'Grade'] = int(0)
        processed_df.loc[raw_df['G3'] >= 10,'Grade'] = int(1)
    elif data_str == 'oulad':
        binary = ['Sex','Disability']
        categorical = ['Region','CodeModule','CodePresentation','HighestEducation','IMDBand']
        numerical = ['NumPrevAttempts','StudiedCredits','AgeGroup']
        label = ['Grade']
        carla_categorical = binary + categorical + ['AgeGroup']
        carla_continuous = ['NumPrevAttempts','StudiedCredits']
        cols = binary + numerical + categorical + label
        raw_df = pd.read_csv(dataset_dir+'oulad/oulad.csv')
        raw_df = erase_missing(raw_df)
        processed_df = pd.DataFrame(index = raw_df.index)
        processed_df.loc[raw_df['gender'] == 'M','Sex'] = 1
        processed_df.loc[raw_df['gender'] == 'F','Sex'] = 2
        processed_df.loc[raw_df['disability'] == 'N','Disability'] = 1
        processed_df.loc[raw_df['disability'] == 'Y','Disability'] = 2
        processed_df.loc[raw_df['region'] == 'East Anglian Region','Region'] = 1
        processed_df.loc[raw_df['region'] == 'Scotland','Region'] = 2
        processed_df.loc[raw_df['region'] == 'North Western Region','Region'] = 3
        processed_df.loc[raw_df['region'] == 'South East Region','Region'] = 4
        processed_df.loc[raw_df['region'] == 'West Midlands Region','Region'] = 5
        processed_df.loc[raw_df['region'] == 'Wales','Region'] = 6
        processed_df.loc[raw_df['region'] == 'North Region','Region'] = 7
        processed_df.loc[raw_df['region'] == 'South Region','Region'] = 8
        processed_df.loc[raw_df['region'] == 'Ireland','Region'] = 9
        processed_df.loc[raw_df['region'] == 'South West Region','Region'] = 10
        processed_df.loc[raw_df['region'] == 'East Midlands Region','Region'] = 11
        processed_df.loc[raw_df['region'] == 'Yorkshire Region','Region'] = 12
        processed_df.loc[raw_df['region'] == 'London Region','Region'] = 13
        processed_df.loc[raw_df['code_module'] == 'AAA','CodeModule'] = 1
        processed_df.loc[raw_df['code_module'] == 'BBB','CodeModule'] = 2
        processed_df.loc[raw_df['code_module'] == 'CCC','CodeModule'] = 3
        processed_df.loc[raw_df['code_module'] == 'DDD','CodeModule'] = 4
        processed_df.loc[raw_df['code_module'] == 'EEE','CodeModule'] = 5
        processed_df.loc[raw_df['code_module'] == 'FFF','CodeModule'] = 6
        processed_df.loc[raw_df['code_module'] == 'GGG','CodeModule'] = 7
        processed_df.loc[raw_df['code_presentation'] == '2013J','CodePresentation'] = 1
        processed_df.loc[raw_df['code_presentation'] == '2014J','CodePresentation'] = 2
        processed_df.loc[raw_df['code_presentation'] == '2013B','CodePresentation'] = 3
        processed_df.loc[raw_df['code_presentation'] == '2014B','CodePresentation'] = 4
        processed_df.loc[raw_df['highest_education'] == 'No Formal quals','HighestEducation'] = 1
        processed_df.loc[raw_df['highest_education'] == 'Post Graduate Qualification','HighestEducation'] = 2
        processed_df.loc[raw_df['highest_education'] == 'Lower Than A Level','HighestEducation'] = 3
        processed_df.loc[raw_df['highest_education'] == 'A Level or Equivalent','HighestEducation'] = 4
        processed_df.loc[raw_df['highest_education'] == 'HE Qualification','HighestEducation'] = 5
        processed_df.loc[(raw_df['imd_band'] == '0-10%') | (raw_df['imd_band'] == '10-20'),'IMDBand'] = 1
        processed_df.loc[(raw_df['imd_band'] == '20-30%') | (raw_df['imd_band'] == '30-40%'),'IMDBand'] = 2
        processed_df.loc[(raw_df['imd_band'] == '40-50%') | (raw_df['imd_band'] == '50-60%'),'IMDBand'] = 3
        processed_df.loc[(raw_df['imd_band'] == '60-70%') | (raw_df['imd_band'] == '70-80%'),'IMDBand'] = 4
        processed_df.loc[(raw_df['imd_band'] == '80-90%') | (raw_df['imd_band'] == '90-100%'),'IMDBand'] = 5
        processed_df.loc[raw_df['age_band'] == '0-35','AgeGroup'] = 1
        processed_df.loc[raw_df['age_band'] == '35-55','AgeGroup'] = 2
        processed_df.loc[raw_df['age_band'] == '55<=','AgeGroup'] = 3
        processed_df['NumPrevAttempts'] = raw_df['num_of_prev_attempts'].astype(int)
        processed_df['StudiedCredits'] = raw_df['studied_credits'].astype(int)
        processed_df.loc[raw_df['final_result'] == 'Fail','Grade'] = int(0)
        processed_df.loc[raw_df['final_result'] == 'Withdrawn','Grade'] = int(0)
        processed_df.loc[raw_df['final_result'] == 'Pass','Grade'] = int(1)
        processed_df.loc[raw_df['final_result'] == 'Distinction','Grade'] = int(1)
    elif data_str == 'law':
        binary = ['WorkFullTime','Sex']
        categorical = ['FamilyIncome','Tier','Race']
        numerical = ['Decile1stYear','Decile3rdYear','LSAT','UndergradGPA','FirstYearGPA','CumulativeGPA']
        label = ['BarExam']
        carla_categorical = binary + categorical
        carla_continuous = numerical
        cols = binary + numerical + categorical + label
        raw_df = pd.read_csv(dataset_dir+'law/law.csv')
        raw_df = erase_missing(raw_df)
        processed_df = pd.DataFrame(index = raw_df.index)
        processed_df['Decile1stYear'] = raw_df['decile1b'].astype(int)
        processed_df['Decile3rdYear'] = raw_df['decile3'].astype(int)
        processed_df['LSAT'] = raw_df['lsat']
        processed_df['UndergradGPA'] = raw_df['ugpa']
        processed_df['FirstYearGPA'] = raw_df['zfygpa']
        processed_df['CumulativeGPA'] = raw_df['zgpa']
        processed_df['WorkFullTime'] = raw_df['fulltime'].astype(int)
        processed_df['FamilyIncome'] = raw_df['fam_inc'].astype(int)
        processed_df.loc[raw_df['male'] == 0.0,'Sex'] = 2
        processed_df.loc[raw_df['male'] == 1.0,'Sex'] = 1
        processed_df['Tier'] = raw_df['tier'].astype(int)
        processed_df['Race'] = raw_df['race'].astype(int)
        processed_df.loc[(raw_df['race'] == 1.0) | (raw_df['race'] == 2.0) | (raw_df['race'] == 3.0) | (raw_df['race'] == 4.0) | (raw_df['race'] == 5.0) | (raw_df['race'] == 6.0) | (raw_df['race'] == 8.0),'Race'] = 2
        processed_df.loc[raw_df['race'] == 7.0,'Race'] = 1
        processed_df['BarExam'] = raw_df['pass_bar'].astype(int)

    data_obj = Dataset(seed,train_fraction,data_str,label,
                 processed_df,binary,categorical,numerical,step,
                 carla_categorical,carla_continuous)
    
    if path_here is not None:
        model_obj = Model(data_obj,path_here)
        data_obj.filter_undesired_class(model_obj)
        data_obj.store_test_undesired()
        data_obj.change_targets_to_numpy()
        return data_obj, model_obj
    else:
        return data_obj
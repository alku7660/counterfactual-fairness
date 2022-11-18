"""
Dataset loader
"""

"""
Imports
"""
import pickle
from model_params import clf_model, best_model_params
from dataset_parameters import define_all_parameters
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
from support import dataset_dir

def euclidean(x1,x2):
    """
    DESCRIPTION:    Calculates the euclidean distance between two different instances
    
    INPUT:
    x1:             Instance 1
    x2:             Instance 2
    
    OUTPUT:
    distance:       Euclidean distance between x1 and x2
    """
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

def sort_data_distance(x, data, data_label):
    """
    DESCRIPTION:    Organize dataset with respect to distance to instance x
    
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
    DESCRIPTION:        Dataset Class

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
                 raw_df, binary, categorical, ordinal, continuous,
                 step, carla_categorical, carla_continuous):

        self.seed = seed_int
        self.train_fraction = train_fraction
        self.name = data_str
        self.label_str = label_str
        self.raw_df = raw_df
        self.raw_df_cols = raw_df.columns
        self.binary = binary
        self.categorical = categorical
        self.ordinal = ordinal
        self.continuous = continuous
        self.numerical = self.ordinal + self.continuous
        self.features = self.binary + self.categorical + self.numerical
        self.bin_enc = None
        self.step = step
        self.carla_categorical = carla_categorical
        self.carla_continuous = carla_continuous
        self.train_df, self.test_df, self.train_target, self.test_target = train_test_split(self.raw_df, self.raw_df[self.label_str], train_size=self.train_fraction, random_state=self.seed)
        self.train_df, self.train_target, self.test_df = self.data_balancing_target_filter() 
        self.bin_enc, self.cat_enc, self.scaler, self.bin_enc_cols, self.cat_enc_cols = self.encoder_scaler_fit()
        self.transformed_train_df, self.transformed_train_np, self.transformed_cols = self.transform_train()
        self.transformed_test_df, self.transformed_test_np = self.transform_test(self.test_df)
        self.carla_enc, self.carla_scaler, self.carla_enc_cols = self.carla_encoder_scaler_fit()
        self.carla_transformed_train_df, self.carla_transformed_train_np, self.carla_transformed_test_df, self.carla_transformed_test_np, self.carla_transformed_cols = self.carla_transform_train_test()
        self.undesired_class = self.undesired_class_data()
        self.define_all_dataset_parameters()
        self.train_sorted = None

    def data_balancing_target_filter(self):
        """
        DESCRIPTION:    Balances the training dataset (Adapted from MACE algorithm methodology - please see: https://github.com/amirhk/mace)
        
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
        train_target = train_df[self.label_str]
        test_df = self.test_df.copy()
        del train_df[self.label_str[0]]
        del test_df[self.label_str[0]]
        return train_df, train_target, test_df

    def encoder_scaler_fit(self):
        """
        DESCRIPTION:    Fits the encoder and scaler for the dataset
        
        INPUT:
        self

        OUTPUT:
        bin_enc:        Fitted binary encoder
        cat_enc:        Fitted categorical encoder
        scaler:         Fitted scaler
        bin_enc_cols:   Binary encoded feature names
        cat_enc_cols:   Categorical encoded feature names
        """
        bin_enc = OneHotEncoder(drop='if_binary', dtype=np.uint8, handle_unknown='error')
        cat_enc = OneHotEncoder(drop='if_binary', dtype=np.uint8, handle_unknown='error')
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
        DESCRIPTION:            Fits the encoder and scaler for the dataset and processes the training dataset

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
        enc_train_data_cat_df = pd.DataFrame(enc_train_data_cat, index=train_data_cat.index, columns=self.cat_enc_cols)
        transformed_train_df = pd.concat((enc_train_data_bin_df, enc_train_data_cat_df, scaled_train_data_num_df),axis=1)
        transformed_cols = transformed_train_df.columns.to_list()
        transformed_train_np = transformed_train_df.to_numpy()
        return transformed_train_df, transformed_train_np, transformed_cols

    def carla_encoder_scaler_fit(self):
        """
        DESCRIPTION:        Fits the encoder and scaler for the dataset and transforms the training dataset according to the CARLA framework

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
        carla_enc_cols = carla_enc.get_feature_names_out(self.carla_categorical)
        return carla_enc, carla_scaler, carla_enc_cols

    def carla_transform_train_test(self):
        """
        DESCRIPTION:                Fits the encoder and scaler for the dataset and transforms the training dataset according to the CARLA framework

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
        carla_transformed_train_df = pd.concat((scaled_carla_train_data_cont_df, enc_carla_train_data_cat_df),axis=1)
        carla_transformed_train_np = carla_transformed_train_df.to_numpy()
        carla_transformed_cols = carla_transformed_train_df.columns.to_list()

        carla_test_data_cat, carla_test_data_cont = self.test_df[self.carla_categorical], self.test_df[self.carla_continuous]
        enc_carla_test_data_cat = self.carla_enc.transform(carla_test_data_cat)
        scaled_carla_test_data_cont = self.carla_scaler.transform(carla_test_data_cont)
        enc_carla_test_data_cat_df = pd.DataFrame(enc_carla_test_data_cat, index=carla_test_data_cat.index, columns=self.carla_enc_cols)
        scaled_carla_test_data_cont_df = pd.DataFrame(scaled_carla_test_data_cont, index=carla_test_data_cont.index, columns=self.carla_continuous)
        carla_transformed_test_df = pd.concat((scaled_carla_test_data_cont_df, enc_carla_test_data_cat_df), axis=1)
        carla_transformed_test_np = carla_transformed_test_df.to_numpy()
        return carla_transformed_train_df, carla_transformed_train_np, carla_transformed_test_df, carla_transformed_test_np, carla_transformed_cols
        
    def transform_test(self, df):
        """
        DESCRIPTION:                Uses the encoder and scaler for the dataset and processes the testing dataset

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
        enc_data_cat_df = pd.DataFrame(enc_data_cat,index=data_cat.index,columns=self.cat_enc_cols)
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
        DESCRIPTION:                        Obtains the undesired class instances according to the selected model
        
        INPUT:
        model:                              Model object containing the trained models
        
        OUTPUT: (None: stored as class attributes)
        undesired_test_df:                  DataFrame containing the original instances with the undesired predicted label
        undesired_test_np:                  Numpy array containing the original instances with the undesired predicted label
        undesired_transformed_test_df:      DataFrame containing the original instances with the undesired predicted label
        undesired_transformed_test_np:      Numpy array containing the original instances with the undesired predicted label
        undesired_test_target:              Ground truth label of the instances predicted with the undesired label
        """
        undesired_test_df = self.test_df.copy()
        undesired_transformed_test_df = self.transformed_test_df.copy()
        undesired_test_df['pred'] = model.sel.predict(self.transformed_test_df)
        undesired_test_target = self.test_target.loc[undesired_test_df['pred'] == self.undesired_class]
        undesired_transformed_test_df = undesired_transformed_test_df.loc[undesired_test_df['pred'] == self.undesired_class]
        undesired_test_df = undesired_test_df.loc[undesired_test_df['pred'] == self.undesired_class]
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
        DESCRIPTION:    Changes the targets to numpy if they are dataframes
        
        INPUT:
        self

        OUTPUT: (None: stored as class attributes)
        Update on:
            train_target
            test_target
            undesired_test_target
        """
        if isinstance(self.train_target, pd.Series) or isinstance(self.train_target, pd.DataFrame):
            self.train_target = self.train_target.to_numpy().reshape((len(self.train_target.to_numpy()),))
        if isinstance(self.test_target, pd.Series) or isinstance(self.test_target, pd.DataFrame):
            self.test_target = self.test_target.to_numpy().reshape((len(self.test_target.to_numpy()),))
        if isinstance(self.undesired_test_target, pd.Series) or isinstance(self.undesired_test_target, pd.DataFrame):
            self.undesired_test_target = self.undesired_test_target.to_numpy().reshape((len(self.undesired_test_target.to_numpy()),))

    def add_test_predictions(self, predictions):
        """
        DESCRIPTION:    Add the test data predictions from a model

        INPUT:
        predictions:    Predictions for the test dataset

        OUTPUT: (None: stored as class attributes)
        test_pred:      Attribute storing the predictions for the test dataset
        """
        self.test_pred = predictions
    
    def add_sorted_train_data(self, instance):
        """
        DESCRIPTION:    Add/change a sorted array of the training dataset according to distance from an instance
        
        INPUT:
        instance:       Instance of interest from which to calculate all the distances

        OUTPUT: (None: stored as class attributes)
        train_sorted:   Sorted training dataset w.r.t. distance to the instance of interest
        """
        start_time = time.time()
        self.train_sorted = sort_data_distance(instance, self.transformed_train_np, self.train_target) 
        end_time = time.time()
        total_time = end_time - start_time
        self.training_sort_time = total_time

    def undesired_class_data(self):
        """
        DESCRIPTION:        Method to obtain the undesired class

        INPUT:
        self

        OUTPUT:
        undesired_class:    Undesired class for the dataset
        """
        if self.name in ['german','credit','compass']:
            undesired_class = 1
        elif self.name in ['adult','kdd_census','dutch','bank','diabetes','student','oulad','law']:
            undesired_class = 0
        return undesired_class

    def from_carla(self, df_instance):
        """
        DESCRIPTION:        Transform from the CARLA instance format to the normal instance format
        
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
        bin_data, cat_data, num_data = df[self.binary], df[self.categorical], df[self.numerical]
        enc_bin_data, enc_cat_data = self.bin_enc.transform(bin_data).toarray(), self.cat_enc.transform(cat_data).toarray()
        enc_bin_data = pd.DataFrame(enc_bin_data,index=bin_data.index, columns=self.bin_enc_cols)
        enc_cat_data = pd.DataFrame(enc_cat_data,index=cat_data.index, columns=self.cat_enc_cols)
        scaled_num_data = pd.DataFrame(self.scaler.transform(num_data), index=num_data.index, columns=self.numerical)
        normal_instance = pd.concat((enc_bin_data, enc_cat_data, scaled_num_data),axis=1)
        return normal_instance

    def store_test_undesired(self):
        """
        DESCRIPTION:        Stores the test_data_undesired information

        INPUT:
        self

        OUTPUT: (None: Storing data to files)
        """
        pickle.dump(self.undesired_test_df, open(f'{dataset_dir}{self.name}/{self.name}_undesired_test_df.pkl', 'wb'))

    def define_all_dataset_parameters(self):
        """
        DESCRIPTION:        Defines all the parameters for the given dataset

        INPUT:
        self

        OUTPUT: (None: stored as class attributes)
        """
        self.feat_type, self.feat_protected, self.feat_mutable, self.feat_dir, self.feat_cost, self.feat_step, self.feat_cat = define_all_parameters(self)

class Model:
    """
    Class that contains the trained models
    DESCRIPTION:        Model class

    INPUT:
    data_obj:           Dataset object
    grid_search_path:   Path to the grid search results
    """
    def __init__(self, data_obj, grid_search_path):
        self.model_params_path = grid_search_path
        self.train_clf_model(data_obj)
    
    def train_clf_model(self, data_obj):
        """
        DESCRIPTION:    Method that trains the classifier model according to the data object received

        INPUT:
        data_obj:           Dataset object
        """
        grid_search_results = pd.read_csv(str(self.model_params_path)+'/Results/grid_search/grid_search_final.csv', index_col = ['dataset','model'])
        sel_model_str, params_best, params_rf = best_model_params(grid_search_results, data_obj.name)
        self.sel, self.rf = clf_model(sel_model_str, params_best, params_rf, data_obj.transformed_train_df, data_obj.train_target)
        self.carla_sel, self.carla_rf = clf_model(sel_model_str, params_best, params_rf, data_obj.carla_transformed_train_df, data_obj.train_target)

def verify_column(df):
    """
    DESCRIPTION:        Verifies whether a column has equal values. If yes, eliminates it from the dataset
    
    INPUT:
    df:                 Dataset as DataFrame
    
    OUTPUT:
    df:                 Dataset modified if a column is found to have all values equal
    """
    for i in df.columns:
        if len(df[i].unique().tolist()) == 1:
            del df[i]
    return df  

def eliminate_columns(df):
    """
    DESCRIPTION:        Eliminates to-be-erased columns from the datasets
    
    INPUT:
    df:                 The training dataset to encode the categorical features as DataFrame
    
    OUTPUT:
    df:                 Dataset with eliminated columns
    """
    for i in df.columns:
        if i.find('to be erased') != -1 or i.find('to be deleted') != -1 or len(df[i].unique()) == 1:
            df.drop(columns=i,inplace=True)
    return df

def erase_duplicates(df):
    """
    DESCRIPTION:        Eliminates duplicate instances
    
    INPUT:
    df:                 The dataset of interest as DataFrame 
    
    OUTPUT:
    df:                 Filtered dataset without duplicate instances
    """
    label = df['class']
    df.drop(columns=['class'],inplace=True)
    df.drop_duplicates(inplace = True)
    df['class'] = label
    df.reset_index(inplace = True)
    df.drop(columns=['index'],inplace=True)
    return df

def nom_to_num(df):
    """
    DESCRIPTION:        Transforms categorical features into encoded numerical values.
    
    INPUT:
    df:                 The dataset to encode the categorical features.
    
    OUTPUT:
    df:                 The dataset with categorical features encoded into numerical features.
    """
    encoder = LabelEncoder()
    if df['label'].dtypes == object or df['label'].dtypes == str:
        encoder.fit(df['label'])
        df['label'] = encoder.transform(df['label'])
    return df, encoder

def load_model_dataset(data_str, train_fraction, seed, step, path_here = None):
    """
    DESCRIPTION:        Loads all datasets according to data_str and train_fraction, and the corresponding selected models for counterfactual search
                        For details about the preparation of each dataset, check the 'data_preparation.py' file. The datasets are here loaded with this prepared (processed) files.
    
    INPUT:
    data_str:           Name of the dataset to load
    train_fraction:     Percentage of dataset instances to use as training dataset
    seed:               Random seed to be used
    step:               Size of the step to be used for continuous variable changes
    path:               Path to the grid search results for model parameter selection

    OUTPUT:
    data_obj:           Dataset object
    model_obj:          Model object
    """
    if data_str == 'adult':
        binary = ['Sex','NativeCountry','Race']
        categorical = ['WorkClass','MaritalStatus','Occupation','Relationship']
        ordinal = ['EducationLevel','AgeGroup']
        continuous = ['EducationNumber','CapitalGain','CapitalLoss','HoursPerWeek']
        label = ['label']
        carla_categorical = ['Sex','AgeGroup','Race','NativeCountry','WorkClass','MaritalStatus','Occupation','Relationship']
        carla_continuous = ['EducationNumber','CapitalGain','CapitalLoss','HoursPerWeek','EducationLevel']
        processed_df = pd.read_csv(dataset_dir+'adult/preprocessed_adult.csv', index_col=0)
    elif data_str == 'kdd_census':
        binary = ['Sex','Race']
        categorical = ['Industry','Occupation']
        ordinal = []
        continuous = ['Age','WageHour','CapitalGain','CapitalLoss','Dividends','WorkWeeksYear']
        label = ['Label']
        carla_categorical = ['Sex','Race','Industry','Occupation']
        carla_continuous = ['Age','WageHour','CapitalGain','CapitalLoss','Dividends','WorkWeeksYear']
        processed_df = pd.read_csv(dataset_dir+'kdd_census/preprocessed_kdd_census.csv', index_col=0)
    elif data_str == 'german':
        binary = ['Sex','Single','Unemployed']
        categorical = ['PurposeOfLoan','InstallmentRate','Housing']
        ordinal = []
        continuous = ['Age','Credit','LoanDuration']
        label = ['Label']
        carla_categorical = ['Sex','Single','Unemployed','PurposeOfLoan','InstallmentRate','Housing']
        carla_continuous = ['Age','Credit','LoanDuration']
        processed_df = pd.read_csv(dataset_dir+'german/preprocessed_german.csv', index_col=0)
    elif data_str == 'dutch':
        binary = ['Sex']
        categorical = ['HouseholdPosition','HouseholdSize','Country','EconomicStatus','CurEcoActivity','MaritalStatus']
        ordinal = ['EducationLevel']
        continuous = ['Age']
        label = ['Occupation']
        carla_categorical = ['Sex','HouseholdPosition','HouseholdSize','Country','EconomicStatus','CurEcoActivity','MaritalStatus','EducationLevel']
        carla_continuous = ['Age']
        processed_df = pd.read_csv(dataset_dir+'dutch/preprocessed_dutch.csv', index_col=0)
    elif data_str == 'bank':
        binary = ['Default','Housing','Loan']
        categorical = ['Job','MaritalStatus','Education','Contact','Month','Poutcome']
        ordinal = ['AgeGroup']
        continuous = ['Balance','Day','Duration','Campaign','Pdays','Previous']
        label = ['Subscribed']
        carla_categorical = ['Default','Housing','Loan','Job','MaritalStatus','Education','Contact','Month','Poutcome','AgeGroup']
        carla_continuous = ['Balance','Day','Duration','Campaign','Pdays','Previous']
        processed_df = pd.read_csv(dataset_dir+'bank/preprocessed_bank.csv', index_col=0)
    elif data_str == 'credit':
        binary = ['isMale','isMarried','HasHistoryOfOverduePayments']
        categorical = []
        ordinal = ['TotalOverdueCounts','TotalMonthsOverdue','AgeGroup','EducationLevel']
        continuous = ['MaxBillAmountOverLast6Months','MaxPaymentAmountOverLast6Months','MonthsWithZeroBalanceOverLast6Months',
                'MonthsWithLowSpendingOverLast6Months','MonthsWithHighSpendingOverLast6Months','MostRecentBillAmount',
                'MostRecentPaymentAmount']
        label = ['NoDefaultNextMonth (label)']
        carla_categorical = ['isMale','isMarried','HasHistoryOfOverduePayments','AgeGroup','EducationLevel']
        carla_continuous = ['MaxBillAmountOverLast6Months','MaxPaymentAmountOverLast6Months','MonthsWithZeroBalanceOverLast6Months',
                'MonthsWithLowSpendingOverLast6Months','MonthsWithHighSpendingOverLast6Months','MostRecentBillAmount',
                'MostRecentPaymentAmount','TotalOverdueCounts','TotalMonthsOverdue']
        processed_df = pd.read_csv(dataset_dir+'/credit/preprocessed_credit.csv')
    elif data_str == 'compass':
        # Based on the MACE algorithm Datasets preprocessing (please, see: https://github.com/amirhk/mace)
        processed_df = pd.DataFrame()
        binary = ['Race','Sex','ChargeDegree']
        categorical = []
        ordinal = ['PriorsCount','AgeGroup']
        continuous = []
        label = ['TwoYearRecid (label)']
        carla_categorical = ['Race','Sex','ChargeDegree','AgeGroup']
        carla_continuous = ['PriorsCount']
        processed_df = pd.read_csv(dataset_dir+'/compass/preprocessed_compass.csv')
    elif data_str == 'diabetes':
        binary = ['DiabetesMed']
        categorical = ['Race','Sex','A1CResult','Metformin','Chlorpropamide','Glipizide','Rosiglitazone','Acarbose','Miglitol']
        ordinal = ['AgeGroup']
        continuous = ['TimeInHospital','NumProcedures','NumMedications','NumEmergency']
        label = ['Label']
        carla_categorical = ['Race','Sex','A1CResult','Metformin','Chlorpropamide','Glipizide','Rosiglitazone','Acarbose','Miglitol','DiabetesMed','AgeGroup']
        carla_continuous = ['TimeInHospital','NumProcedures','NumMedications','NumEmergency']
        processed_df = pd.read_csv(dataset_dir+'/diabetes/preprocessed_diabetes.csv')
    elif data_str == 'student':
        binary = ['School','Sex','AgeGroup','Address','FamilySize','ParentStatus','SchoolSupport','FamilySupport','ExtraPaid','ExtraActivities','Nursery','HigherEdu','Internet','Romantic']
        categorical = ['MotherJob','FatherJob','SchoolReason']
        ordinal = ['MotherEducation','FatherEducation']
        continuous = ['TravelTime','ClassFailures','GoOut']
        label = ['Grade']
        carla_categorical = binary + categorical
        carla_continuous = ['MotherEducation','FatherEducation','TravelTime','ClassFailures','GoOut']
        processed_df = pd.read_csv(dataset_dir+'/student/preprocessed_student.csv')
    elif data_str == 'oulad':
        binary = ['Sex','Disability']
        categorical = ['Region','CodeModule','CodePresentation','HighestEducation','IMDBand']
        ordinal = ['AgeGroup']
        continuous = ['NumPrevAttempts','StudiedCredits']
        label = ['Grade']
        carla_categorical = binary + categorical + ordinal
        carla_continuous = ['NumPrevAttempts','StudiedCredits']
        processed_df = pd.read_csv(dataset_dir+'/oulad/preprocessed_oulad.csv')
    elif data_str == 'law':
        binary = ['WorkFullTime','Sex']
        categorical = ['FamilyIncome','Tier','Race']
        ordinal = []
        continuous = ['Decile1stYear','Decile3rdYear','LSAT','UndergradGPA','FirstYearGPA','CumulativeGPA']
        label = ['BarExam']
        carla_categorical = binary + categorical
        carla_continuous = continuous
        processed_df = pd.read_csv(dataset_dir+'/law/preprocessed_law.csv')

    data_obj = Dataset(seed, train_fraction, data_str, label,
                 processed_df, binary, categorical, ordinal, continuous,
                 step, carla_categorical, carla_continuous)
    
    if path_here is not None:
        model_obj = Model(data_obj, path_here)
        data_obj.filter_undesired_class(model_obj)
        data_obj.store_test_undesired()
        data_obj.change_targets_to_numpy()
        return data_obj, model_obj
    else:
        return data_obj
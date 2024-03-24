import numpy as np
import pandas as pd
import copy
from address import dataset_dir
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from mlxtend.preprocessing import TransactionEncoder
import time

def euclidean(x1,x2):
    """
    Calculates the euclidean distance between two different instances
    """
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

def sort_data_distance(x, data, data_label):
    """
    Organize dataset with respect to distance to instance x
    """
    sort_data_distance = []
    for i in range(len(data)):
        dist = euclidean(data[i],x)
        sort_data_distance.append((data[i],dist,data_label[i]))      
    sort_data_distance.sort(key=lambda x: x[1])
    return sort_data_distance

class Dataset:

    def __init__(self, data_str, seed_int, train_fraction, label_str,
                 df, binary, categorical, ordinal, continuous,
                 step) -> None:
        self.name = data_str
        self.seed = seed_int
        self.train_fraction = train_fraction
        self.label_name = label_str
        self.binary = binary
        self.categorical = categorical
        self.ordinal = ordinal
        self.continuous = continuous
        self.numerical = self.ordinal + self.continuous
        self.features = binary + categorical + ordinal + continuous
        self.df = df[self.features + self.label_name]
        self.step = step
        self.train_df, self.test_df, self.train_target, self.test_target = train_test_split(self.df, self.df[self.label_name], train_size=self.train_fraction, random_state=self.seed)
        self.train_df, self.train_target = self.balance_train_data()
        self.bin_cat_ord_enc = self.all_one_hot_encode(self.train_df)
        if len(self.continuous) > 0:
            self.discretizer = self.train_discretizer(self.train_df)
        self.discretized_train_df = self.discretize_df(self.train_df)
        self.discretized_test_df = self.discretize_df(self.test_df)
        self.bin_enc, self.cat_enc, self.bin_cat_enc, self.scaler = self.encoder_scaler_fit()
        self.bin_enc_cols, self.cat_enc_cols, self.bin_cat_enc_cols = self.encoder_scaler_cols()
        self.processed_features = list(self.bin_enc_cols) + list(self.cat_enc_cols) + self.ordinal + self.continuous
        self.processed_features_dict_idx = self.get_idx_processed_features_dict()
        self.processed_ordinal_continuous_idx_list = self.get_idx_ordinal_continuous_processed_features()
        self.transformed_train_df = self.transform_data(self.train_df)
        self.transformed_train_np = self.transformed_train_df.to_numpy()
        self.transformed_test_df = self.transform_data(self.test_df)
        self.transformed_test_np = self.transformed_test_df.to_numpy()
        self.train_target, self.test_target = self.change_targets_to_numpy()
        self.undesired_class = self.undesired_class_data()
        self.desired_class = int(1 - self.undesired_class)
        self.feat_type = self.define_feat_type()
        self.feat_protected = self.define_protected()
        self.feat_mutable = self.define_mutable()
        self.immutables = self.get_immutables()
        self.feat_dir = self.define_feat_directionality()
        self.feat_step = self.define_feat_step()
        self.feat_cat = self.define_feat_cat()
        self.idx_cat_cols_dict = self.idx_cat_columns()
        self.feat_dist, self.processed_feat_dist = self.feature_distribution()

    def get_idx_processed_features_dict(self):
        """
        Gets the indices of the processed features, and stores it as a dictionary
        """
        processed_features_idx_dict = {}
        bin_cat_list = list(self.bin_enc_cols) + list(self.cat_enc_cols)
        ord_con_list = self.ordinal + self.continuous
        for feature in self.features:
            feature_idx_list = []
            for processed_feature_idx in range(len(bin_cat_list)):
                processed_feature = bin_cat_list[processed_feature_idx]
                if f'{feature}_' in processed_feature:
                    feature_idx_list.extend([processed_feature_idx])
            for processed_feature_idx in range(len(bin_cat_list),len(self.processed_features)):
                processed_feature = ord_con_list[processed_feature_idx-len(bin_cat_list)]
                if f'{feature}' in processed_feature:
                    feature_idx_list.extend([processed_feature_idx])
            processed_features_idx_dict[feature] = feature_idx_list
        return processed_features_idx_dict

    def get_idx_ordinal_continuous_processed_features(self):
        """
        Gets the indices of the ordinal and continuous processed features
        """
        processed_ordinal_continuous_idx_list = []
        for ordinal_continuous_feature in self.ordinal+self.continuous:
            processed_ordinal_continuous_idx_list.extend(self.processed_features_dict_idx[ordinal_continuous_feature])
        return processed_ordinal_continuous_idx_list

    def balance_train_data(self):
        """
        Method to balance the training dataset using undersampling of majority class
        """
        train_data_label = self.train_df[self.label_name]
        label_value_counts = train_data_label.value_counts()
        samples_per_class = label_value_counts.min()
        balanced_train_df = pd.concat([self.train_df[(train_data_label == 0).to_numpy()].sample(samples_per_class, random_state = self.seed),
        self.train_df[(train_data_label == 1).to_numpy()].sample(samples_per_class, random_state = self.seed),]).sample(frac = 1, random_state = self.seed)
        balanced_train_df_label = balanced_train_df[self.label_name]
        try:
            del balanced_train_df[self.label_name]
        except:
            del balanced_train_df[self.label_name[0]]
        return balanced_train_df, balanced_train_df_label
    
    def train_discretizer(self, df, bins = 6):
        """
        Obtains a discretizer for the continuous features
        """
        cont_df = df[self.continuous]
        discretizer = KBinsDiscretizer(n_bins=bins)
        discretizer.fit(cont_df)
        return discretizer

    def discretize_continuous_feat(self, cont_df):
        """
        Makes all continuous features categorical
        """
        discretized_np = self.discretizer.transform(cont_df)
        discretized_cols = self.discretizer.get_feature_names_out(cont_df.columns)
        discretized_df = pd.DataFrame(index=cont_df.index, data=discretized_np.toarray(), columns=discretized_cols)
        return discretized_df

    def all_one_hot_encode(self, df):
        """
        Obtains a fully-one-hot-encoder for the discretization of the data
        """
        bin_cat_ord_enc = OneHotEncoder(dtype=np.uint8, handle_unknown='ignore')
        bin_cat_ord_df = df[self.binary + self.categorical + self.ordinal]
        bin_cat_ord_df = bin_cat_ord_df.round(0).astype(int)
        bin_cat_ord_enc.fit(bin_cat_ord_df)
        return bin_cat_ord_enc

    def discretize_df(self, df):
        """
        Obtains a fully-one-hot-encoded version of the input df (DataFrame) for the apriori algorithm
        """
        cont_df = df[self.continuous]
        if len(self.continuous) > 0:
            discretized_cont_df = self.discretize_continuous_feat(cont_df)
        else:
            discretized_cont_df = pd.DataFrame()
        bin_cat_ord_df = df[self.binary + self.categorical + self.ordinal]
        try:
            bin_cat_ord_df = bin_cat_ord_df.round(0).astype(int)
        except:
            bin_cat_ord_df = bin_cat_ord_df
        discretized_bin_cat_ord_np = self.bin_cat_ord_enc.transform(bin_cat_ord_df)
        bin_cat_ord_cols = self.bin_cat_ord_enc.get_feature_names_out(self.binary + self.categorical + self.ordinal)
        discretized_bin_cat_ord_df = pd.DataFrame(index=df.index, data=discretized_bin_cat_ord_np.toarray(), columns=bin_cat_ord_cols)
        all_df = pd.concat((discretized_bin_cat_ord_df, discretized_cont_df), axis=1)
        return all_df

    def decode_df(self, encoded_df):
        """
        Performs a the inverse operation of the discretization of the df (uses the encoder and discretizer)
        """
        
        # print(encoded_df.to_frame().T)
        if len(self.continuous) > 0:
            encoded_continuous_cols = self.discretizer.get_feature_names_out(self.continuous)
        else:
            encoded_continuous_cols = []
        bin_cat_ord_cols = self.bin_cat_ord_enc.get_feature_names_out(self.binary + self.categorical + self.ordinal)
        cont_encoded_df = encoded_df[encoded_continuous_cols]
        bin_cat_ord_encoded_df = encoded_df[bin_cat_ord_cols]
        if len(self.continuous) > 0:
            cont_np = self.discretizer.inverse_transform(cont_encoded_df)
        else:
            cont_np = []
        cont_df = pd.DataFrame(index=encoded_df.index, data=cont_np, columns=self.continuous)
        bin_cat_ord_np = self.bin_cat_ord_enc.inverse_transform(bin_cat_ord_encoded_df)
        bin_cat_ord_df = pd.DataFrame(index=encoded_df.index, data=bin_cat_ord_np, columns=self.binary + self.categorical + self.ordinal)
        all_df = pd.concat((bin_cat_ord_df, cont_df), axis=1)
        return all_df

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

    def encoder_scaler_fit(self):
        """
        Method that fits the encoders and scaler for the dataset
        """
        bin_enc = OneHotEncoder(drop='if_binary', dtype=np.uint8, handle_unknown='ignore')
        cat_enc = OneHotEncoder(drop='if_binary', dtype=np.uint8, handle_unknown='ignore')
        bin_cat_enc = OneHotEncoder(drop='if_binary', dtype=np.uint8, handle_unknown='ignore')
        scaler = MinMaxScaler(clip=True)
        bin_enc.fit(self.train_df[self.binary])
        cat_enc.fit(self.train_df[self.categorical])
        bin_cat_enc.fit(self.train_df[self.binary + self.categorical])
        scaler.fit(self.train_df[self.ordinal + self.continuous])
        return bin_enc, cat_enc, bin_cat_enc, scaler

    def encoder_scaler_cols(self):
        """
        Method that extracts the encoded columns from the encoders
        """
        return list(self.bin_enc.get_feature_names_out(self.binary)), list(self.cat_enc.get_feature_names_out(self.categorical)), list(self.bin_cat_enc.get_feature_names_out(self.binary + self.categorical))

    def transform_data(self, data_df):
        """
        Method that transforms the input dataframe using the encoder and scaler
        """
        data_bin, data_cat, data_ord_cont = data_df[self.binary], data_df[self.categorical], data_df[self.ordinal + self.continuous]
        enc_data_bin = self.bin_enc.transform(data_bin).toarray()
        enc_data_cat = self.cat_enc.transform(data_cat).toarray()
        sca_data_ord_cont = self.scaler.transform(data_ord_cont)
        enc_data_bin_df = pd.DataFrame(enc_data_bin, index=data_bin.index, columns=self.bin_enc_cols)
        enc_data_cat_df = pd.DataFrame(enc_data_cat, index=data_cat.index, columns=self.cat_enc_cols)
        sca_data_ord_cont_df = pd.DataFrame(sca_data_ord_cont, index=data_ord_cont.index, columns=self.ordinal+self.continuous)
        transformed_data_df = pd.concat((enc_data_bin_df, enc_data_cat_df, sca_data_ord_cont_df), axis=1)
        return transformed_data_df

    def undesired_class_data(self):
        """
        Method to obtain the undesired class
        """
        if self.name in ['compass','credit','german','heart','synthetic_disease','diabetes']:
            undesired_class = 1
        elif self.name in ['ionosphere','adult','kdd_census','dutch','bank','synthetic_athlete','student','oulad','law']:
            undesired_class = 0
        return undesired_class
    
    def undesired_test(self, model):
        """
        Method to obtain the test subset with predicted undesired class
        """
        self.undesired_transformed_test_df = self.transformed_test_df.loc[model.model.predict(self.transformed_test_np) == self.undesired_class]
        self.undesired_test_target = self.test_target[model.model.predict(self.transformed_test_np) == self.undesired_class]
        self.undesired_transformed_test_np = self.undesired_transformed_test_df.to_numpy()
        desired_ground_truth_transformed_test_df = self.transformed_test_df.loc[self.test_target != self.undesired_class]
        self.desired_ground_truth_test_df = self.test_df.loc[self.test_target != self.undesired_class]
        desired_ground_truth_target = self.test_target[self.test_target != self.undesired_class]
        predicted_label_desired_ground_truth_test_df = model.model.predict(desired_ground_truth_transformed_test_df)
        self.false_undesired_test_df = self.desired_ground_truth_test_df.loc[predicted_label_desired_ground_truth_test_df == self.undesired_class]
        self.transformed_false_undesired_test_df = desired_ground_truth_transformed_test_df.loc[predicted_label_desired_ground_truth_test_df == self.undesired_class]
        self.false_undesired_target = desired_ground_truth_target[predicted_label_desired_ground_truth_test_df == self.undesired_class]
        del self.desired_ground_truth_test_df[self.label_name[0]]
        del self.false_undesired_test_df[self.label_name[0]]

    def define_feat_type(self):
        """
        Method that obtains a feature type vector corresponding to each of the features
        """
        feat_type = copy.deepcopy(self.transformed_train_df.dtypes)
        feat_list = feat_type.index.tolist()
        if self.name == 'adult':
            for i in feat_list:
                if 'Sex' in i or 'Native' in i or 'Race' in i:
                    feat_type.loc[i] = 'bin'
                elif 'WorkClass' in i or 'Marital' in i or 'Occupation' in i or 'Relation' in i:
                    feat_type.loc[i] = 'cat'
                elif 'EducationLevel' in i or 'Age' in i:
                    feat_type.loc[i] = 'ord'
                elif 'EducationNumber' in i or 'Capital' in i or 'Hours' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'kdd_census':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_type.loc[i] = 'bin'
                elif  'Industry' in i or 'Occupation' in i:
                    feat_type.loc[i] = 'cat'
                elif 'Age' in i or 'WageHour' in i or 'Capital' in i or 'Dividends' in i or 'WorkWeeksYear' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'german':
            for i in feat_list:
                if 'Sex' in i or 'Single' in i or 'Unemployed' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Housing' in i or 'PurposeOfLoan' in i or 'InstallmentRate' in i:
                    feat_type.loc[i] = 'cat'
                elif 'Age' in i or 'Credit' in i or 'Loan' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'dutch':
            for i in feat_list:
                if 'Sex' in i:
                    feat_type.loc[i] = 'bin'
                elif 'HouseholdPosition' in i or 'HouseholdSize' in i or 'Country' in i or 'EconomicStatus' in i or 'CurEcoActivity' in i or 'MaritalStatus' in i:
                    feat_type.loc[i] = 'cat'
                elif 'EducationLevel' in i:
                    feat_type.loc[i] = 'ord'
                elif 'Age' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'bank':
            for i in feat_list:
                if 'Default' in i or 'Housing' in i or 'Loan' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Job' in i or 'MaritalStatus' in i or 'Education' in i or 'Contact' in i or 'Month' in i or 'Poutcome' in i:
                    feat_type.loc[i] = 'cat'
                elif 'Age' in i:
                    feat_type.loc[i] = 'ord'
                elif 'Balance' in i or 'Day' in i or 'Duration' in i or 'Campaign' in i or 'Pdays' in i or 'Previous' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'credit':
            for i in feat_list:
                if 'Male' in i or 'Married' in i or 'History' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Total' in i or 'Age' in i or 'Education' in i:
                    feat_type.loc[i] = 'ord'
                elif 'Amount' in i or 'Balance' in i or 'Spending' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'compass':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i or 'Charge' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Priors' in i or 'Age' in i:
                    feat_type.loc[i] = 'ord'
        elif self.name == 'diabetes':
            for i in feat_list:
                if 'DiabetesMed' in i or 'Sex' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Race' in i or 'A1CResult' in i or 'Metformin' in i or 'Chlorpropamide' in i or 'Glipizide' in i or 'Rosiglitazone' in i or 'Acarbose' in i or 'Miglitol' in i:
                    feat_type.loc[i] = 'cat'
                elif 'AgeGroup' in i:
                    feat_type.loc[i] = 'ord'
                elif 'TimeInHospital' in i or 'NumProcedures' in i or 'NumMedications' in i or 'NumEmergency' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'student':
            for i in feat_list:
                if 'Age' in i or 'School' in i or 'Sex' in i or 'Address' in i or 'FamilySize' in i or 'ParentStatus' in i or 'SchoolSupport' in i or 'FamilySupport' in i or 'ExtraPaid' in i or 'ExtraActivities' in i or 'Nursery' in i or 'HigherEdu' in i or 'Internet' in i or 'Romantic' in i:
                    feat_type.loc[i] = 'bin'
                elif 'MotherJob' in i or 'FatherJob' in i or 'SchoolReason' in i:
                    feat_type.loc[i] = 'cat'
                elif 'MotherEducation' in i or 'FatherEducation' in i:
                    feat_type.loc[i] = 'ord'
                elif 'TravelTime' in i or 'ClassFailures' in i or 'GoOut' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'oulad':
            for i in feat_list:
                if 'Sex' in i or 'Disability' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Region' in i or 'CodeModule' in i or 'CodePresentation' in i or 'HighestEducation' in i or 'IMDBand' in i:
                    feat_type.loc[i] = 'cat'
                elif 'NumPrevAttempts' in i or 'StudiedCredits' in i:
                    feat_type.loc[i] = 'cont'
                elif 'AgeGroup' in i:
                    feat_type.loc[i] = 'ord'
        elif self.name == 'law':
            for i in feat_list:
                if 'Race' in i or 'WorkFullTime' in i or 'Sex' in i:
                    feat_type.loc[i] = 'bin'
                elif 'FamilyIncome' in i or 'Tier' in i:
                    feat_type.loc[i] = 'cat'
                elif 'Decile1stYear' in i or 'Decile3rdYear' in i or 'LSAT' in i or 'UndergradGPA' in i or 'FirstYearGPA' in i or 'CumulativeGPA' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'ionosphere':
            for i in feat_list:
                feat_type.loc[i] = 'cont'
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if 'Sex' in i:
                    feat_type.loc[i] = 'bin'
                elif 'TrainingTime' in i or 'Diet' in i or 'Sport' in i:
                    feat_type.loc[i] = 'cat'
                elif 'Age' in i or 'SleepHours' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'heart':
            for i in feat_list:
                if 'Sex' in i or 'BloodSugar' in i or 'ChestPain' in i:
                    feat_type.loc[i] = 'bin'
                elif 'ECG' in i:
                    feat_type.loc[i] = 'ord'
                elif 'Age' in i or 'RestBloodPressure' in i or 'Chol' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'synthetic_disease':
            for i in feat_list:
                if 'Smokes' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Diet' in i or 'Stress' in i:
                    feat_type.loc[i] = 'cat'
                elif 'Weight' in i:
                    feat_type.loc[i] = 'ord'
                elif 'Age' in i or 'ExerciseMinutes' in i or 'SleepHours' in i:
                    feat_type.loc[i] = 'cont'
        return feat_type

    def define_protected(self):
        """
        DESCRIPTION:        Defines which features are sensitive / protected and the groups or categories in each of them
        
        INPUT:
        self:               self object

        OUTPUT:
        feat_protected:     Protected set of features per selfset
        """
        feat_protected = dict()
        if self.name == 'adult':
            feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
            feat_protected['Race'] = {1.00:'White', 2.00:'Non-white'}
            feat_protected['AgeGroup'] = {1.00:'<25', 2.00:'25-60', 3.00:'>60'}
        elif self.name == 'kdd_census':
            feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
            feat_protected['Race'] = {1.00:'White', 2.00:'Non-white'}
        elif self.name == 'german':
            feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
        elif self.name == 'dutch':
            feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
        elif self.name == 'bank':
            feat_protected['AgeGroup'] = {1.00:'<25', 2.00:'25-60', 3.00:'>60'}
            feat_protected['MaritalStatus'] = {1.00:'Married', 2.00:'Single', 3.00:'Divorced'}
        elif self.name == 'credit':
            feat_protected['isMale'] = {1.00:'True', 0.00:'False'}
            feat_protected['isMarried'] = {1.00:'True', 0.00:'False'}
            # feat_protected['AgeGroup'] = {}
            # feat_protected['EducationLevel'] = {1.00:'Other', 2.00:'HS', 3.00:'University', 4.00:'Graduate'}
        elif self.name == 'compass':
            feat_protected['Race'] = {1.00:'African-American', 2.00:'Caucasian'}
            feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
        elif self.name == 'diabetes':
            feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
        elif self.name == 'student':
            feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
            feat_protected['AgeGroup'] = {1.00:'<18', 2.00:'>=18'}
        elif self.name == 'oulad':
            feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
        elif self.name == 'law':
            feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
            feat_protected['Race'] = {1.00:'White', 2.00:'Non-white'}
        elif self.name == 'synthetic_athlete':
            feat_protected['Sex'] = {0.00:'Male', 1.00:'Female'}
        return feat_protected

    def define_mutable(self):
        """
        DESCRIPTION:        Method that outputs mutable features per dataset

        INPUT:
        data:               Data object
        feat_protected:     Dictionary contatining the protected features and the sensitive groups names

        OUTPUT:
        feat_mutable:       Series indicating the mutability of each feature
        """
        feat_list = self.processed_features
        feat_mutable  = dict()
        for i in feat_list:
            feat_mutable[i] = 1
        for i in self.feat_protected.keys():
            idx_feat_protected = [j for j in range(len(feat_list)) if i in feat_list[j]]
            feat = [feat_list[j] for j in idx_feat_protected]
            for j in feat:
                feat_mutable[j] = 0
        # if self.name in ['adult','dutch']:
        #     immutable_not_protected = ['MaritalStatus']
        if self.name == 'german':
            immutable_not_protected = ['Single']
        elif self.name == 'student':
            immutable_not_protected = ['ParentStatus']
        elif self.name == 'oulad':
            immutable_not_protected = ['Disability']
        else:
            immutable_not_protected = []
        for feat_immutable_not_protected in immutable_not_protected:
            feat = [j for j in feat_list if feat_immutable_not_protected in j]
            for j in feat:
                feat_mutable[j] = 0
        feat_mutable = pd.Series(feat_mutable)
        return feat_mutable
    
    def get_immutables(self):
        """
        Outputs the immutable features list according to the mutability property
        """
        immutables = []
        for i in self.feat_mutable.keys():
            if self.feat_mutable[i] == 0:
                immutables.append(i)
        return immutables

    def define_feat_directionality(self):
        """
        Method that outputs change directionality of features per dataset
        """
        feat_directionality = copy.deepcopy(self.transformed_train_df.dtypes)
        feat_list = feat_directionality.index.tolist()
        if self.name == 'adult':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'Race' in i or 'Native' in i or 'MaritalStatus' in i:
                    feat_directionality[i] = 0
                elif 'Education' in i:
                    feat_directionality[i] = 'pos'
                else:
                    feat_directionality[i] = 'any'
        elif self.name == 'kdd_census':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_directionality[i] = 0
                elif 'Industry' in i or 'Occupation' in i or 'WageHour' in i or 'CapitalGain' in i or 'CapitalLoss' in i or 'Dividends' in i or 'WorkWeeksYear' or 'Age' in i:
                    feat_directionality[i] = 'any'
        elif self.name == 'german':
            for i in feat_list:
                if 'Sex' in i or 'Single' in i:
                    feat_directionality[i] = 0
                elif 'Age' in i:
                    feat_directionality[i] = 'pos'
                else:
                    feat_directionality[i] = 'any'
        elif self.name == 'dutch':
            for i in feat_list:
                if 'Sex' in i or 'Country' in i or 'MaritalStatus' in i:
                    feat_directionality[i] = 0
                elif 'HouseholdPosition' in i or 'HouseholdSize' in i or 'EconomicStatus' in i or 'CurEcoActivity' in i:
                    feat_directionality[i] = 'any'
                elif 'EducationLevel' in i or 'Age' in i:
                    feat_directionality[i] = 'pos'
        elif self.name == 'bank':
            for i in feat_list:
                if 'Age' in i or 'Marital' in i:
                    feat_directionality[i] = 0
                elif 'Default' in i or 'Housing' in i or 'Loan' in i or 'Job' in i or 'Contact' in i or 'Month' in i or 'Poutcome' or 'Balance' in i or 'Day' in i or 'Duration' in i or 'Campaign' in i or 'Pdays' in i or 'Previous' in i:
                    feat_directionality[i] = 'any'
        elif self.name == 'credit':
            for i in feat_list:
                if 'Male' in i or 'Married' in i:
                    feat_directionality[i] = 0
                elif 'OverLast6Months' in i or 'MostRecent' in i or 'Total' in i or 'History' in i:
                    feat_directionality[i] = 'any'
                if 'Age' in i or 'Education' in i:
                    feat_directionality[i] = 'pos'
        elif self.name == 'compass':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_directionality[i] = 0
                elif 'Age' in i:
                    feat_directionality[i] = 'pos'
                elif 'Charge' in i or 'Priors' in i:
                    feat_directionality[i] = 'any'
        elif self.name == 'diabetes':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_directionality[i] = 0
                elif 'Age' in i:
                    feat_directionality[i] = 'pos'
                else:
                    feat_directionality[i] = 'any'
        elif self.name == 'student':
            for i in feat_list:
                if 'Sex' in i or 'Age' in i or 'ParentStatus' in i:
                    feat_directionality[i] = 0
                elif 'MotherEducation' in i or 'FatherEducation' in i:
                    feat_directionality[i] = 'pos'
                else:
                    feat_directionality[i] = 'any'
        elif self.name == 'oulad':
            for i in feat_list:
                if 'Sex' in i or 'Disability' in i:
                    feat_directionality[i] = 0
                else:
                    feat_directionality[i] = 'any'
        elif self.name == 'law':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_directionality[i] = 0
                else:
                    feat_directionality[i] = 'any'
        elif self.name == 'ionosphere':
            for i in feat_list:
                feat_directionality[i] = 'any'
        elif self.name == 'heart':
            for i in feat_list:
                if 'BloodSugar' in i or 'RestBloodPressure' in i or 'Chol' in i or 'ChestPain' in i or 'ECG' in i:
                    feat_directionality[i] = 'any'
                elif 'Sex' in i:
                    feat_directionality[i] = 0
                elif 'Age' in i:
                    feat_directionality[i] = 'pos'
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if 'Sex' in i:
                    feat_directionality[i] = 0
                elif 'TrainingTime' in i or 'Diet' in i or 'Sport' in i or 'SleepHours' in i:
                    feat_directionality[i] = 'any'
                elif 'Age' in i:
                    feat_directionality[i] = 'pos'
        elif self.name == 'synthetic_disease':
            for i in feat_list:
                if 'Age' in i:
                    feat_directionality[i] = 'pos'
                elif 'ExerciseMinutes' in i or 'SleepHours' in i or 'Weight' in i or 'Diet' in i or 'Stress' in i or 'Smokes' in i:
                    feat_directionality[i] = 'any'
        return feat_directionality

    def define_feat_step(self):
        """
        Method that estimates the step size of all features (used for ordinal features)
        """
        feat_step = pd.Series(data=1/(self.scaler.data_max_ - self.scaler.data_min_), index=[i for i in self.feat_type.keys() if self.feat_type[i] in ['ord','cont']])
        for i in self.feat_type.keys().tolist():
            if self.feat_type.loc[i] == 'cont':
                feat_step.loc[i] = self.step
            elif self.feat_type.loc[i] == 'ord':
                continue
            else:
                feat_step.loc[i] = 0
        feat_step = feat_step.reindex(index = self.feat_type.keys().to_list())
        return feat_step

    def define_feat_cat(self):
        """
        Method that assigns categorical groups to different one-hot encoded categorical features
        """
        feat_cat = copy.deepcopy(self.transformed_train_df.dtypes)
        feat_list = feat_cat.index.tolist()
        if self.name == 'adult':
            for i in feat_list:
                if 'Sex' in i or 'Native' in i or 'EducationLevel' in i or 'EducationNumber' in i or 'Capital' in i or 'Hours' in i or 'Race' in i or 'Age' in i:
                    feat_cat.loc[i] = 'non'
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
                else:
                    feat_cat.loc[i] = 'non'
        elif self.name == 'ionosphere':
            for i in feat_list:
                feat_cat[i] = 'non'
        elif self.name == 'heart':
            for i in feat_list:
                if 'ChestPain' in i:
                    feat_cat[i] = 'cat_1'
                else:
                    feat_cat[i] = 'non' 
        elif self.name == 'synthetic_athlete':
            for i in feat_list:
                if 'Age' in i or 'Sex' in i or 'SleepHours' in i:
                    feat_cat.loc[i] = 'non'
                elif 'TrainingTime' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Diet' in i:
                    feat_cat.loc[i] = 'cat_2'
                elif 'Sport' in i:
                    feat_cat.loc[i] = 'cat_3'
        elif self.name == 'synthetic_disease':
            for i in feat_list:
                if 'Age' in i or 'Smokes' in i or 'ExerciseMinutes' in i or 'SleepHours' in i or 'Weight' in i:
                    feat_cat.loc[i] = 'non'
                elif 'Diet' in i:
                    feat_cat.loc[i] = 'cat_1'
                elif 'Stress' in i:
                    feat_cat.loc[i] = 'cat_2'
        return feat_cat

    def idx_cat_columns(self):
        """
        Method that obtains the indices of the columns of the categorical variables 
        """
        feat_index = range(len(self.processed_features))
        dict_idx_cat = {}
        for i in self.cat_enc_cols:
            if i[:-4] not in list(dict_idx_cat.keys()): 
                cat_cols_idx = [s for s in feat_index if i[:-4] in self.processed_features[s]]
                dict_idx_cat[i[:-4]] = cat_cols_idx
        return dict_idx_cat

    def feature_distribution(self):
        """
        Method to calculate the distribution for all features
        """
        num_instances_train_df = self.train_df.shape[0]
        num_instances_processed_train = self.transformed_train_df.shape[0]
        feat_dist = {}
        processed_feat_dist = {}
        all_non_con_feat = self.binary + self.categorical + self.ordinal
        all_non_con_processed_feat = list(self.bin_enc_cols) + list(self.cat_enc_cols) + self.ordinal
        if len(all_non_con_feat) > 0:
            for i in all_non_con_feat:
                feat_dist[i] = ((self.train_df[i].value_counts()+1)/(num_instances_train_df+len(np.unique(self.train_df[i])))).to_dict() # +1 for laplacian counter
        if len(self.continuous) > 0:
            for i in self.continuous:
                feat_dist[i] = {'mean': self.train_df[i].mean(), 'std': self.train_df[i].std()}
                processed_feat_dist[i] = {'mean': self.transformed_train_df[i].mean(), 'std': self.transformed_train_df[i].std()}
        if len(all_non_con_processed_feat) > 0:
            for i in all_non_con_processed_feat:
                processed_feat_dist[i] = ((self.transformed_train_df[i].value_counts()+1)/(num_instances_processed_train+len(np.unique(self.transformed_train_df[i])))).to_dict() # +1 for laplacian counter
        return feat_dist, processed_feat_dist

    def change_targets_to_numpy(self):
        """
        Method that changes the targets to numpy if they are dataframes
        """
        if isinstance(self.train_target, pd.Series) or isinstance(self.train_target, pd.DataFrame):
            train_target = self.train_target.to_numpy().reshape((len(self.train_target.to_numpy()),))
        if isinstance(self.test_target, pd.Series) or isinstance(self.test_target, pd.DataFrame):
            test_target = self.test_target.to_numpy().reshape((len(self.test_target.to_numpy()),))
        return train_target, test_target

    def inverse(self, normal_x, mace=False):
        """
        Method that transforms an instance back into the original space
        """
        if mace:
            x_df = copy.deepcopy(normal_x)
            for col in self.categorical:
                mace_cat_cols = [i for i in x_df.columns if '_cat_' in i and col in i]
                for mace_col in mace_cat_cols:
                    if x_df[mace_col].values[0] == 1:
                        col_name_value = mace_col.split('_cat_')
                        col_name, col_value = col_name_value[0], int(col_name_value[1]) + 1
                        break
                x_df[col_name] = col_value
                x_df.drop(mace_cat_cols, axis=1, inplace=True)
            for col in self.ordinal:
                mace_ord_cols = [i for i in x_df.columns if '_ord_' in i and col in i]
                current_col_with_1 = 0
                for ord_col in mace_ord_cols:
                    if x_df[ord_col].values[0] == 1:
                        current_col_with_1 = ord_col
                        if mace_ord_cols[-1] == ord_col:
                            col_name_value = current_col_with_1.split('_ord_')
                            col_name, col_value = col_name_value[0], int(col_name_value[1]) + 1
                    elif x_df[ord_col].values[0] == 0:
                        col_name_value = current_col_with_1.split('_ord_')
                        col_name, col_value = col_name_value[0], int(col_name_value[1]) + 1
                        break
                x_df[col_name] = col_value
                x_df.drop(mace_ord_cols, axis=1, inplace=True)
            x = x_df[self.features]                  
        else:
            normal_x_df = pd.DataFrame(data=normal_x.reshape(1, -1), columns=self.processed_features)
            normal_x_df_bin, normal_x_df_cat, normal_x_df_ord_cont = normal_x_df[self.bin_enc_cols], normal_x_df[self.cat_enc_cols], normal_x_df[self.ordinal+self.continuous]
            try:
                x_bin = self.bin_enc.inverse_transform(normal_x_df_bin)
            except:
                x_bin = np.array([[]])
            try:
                x_cat = self.cat_enc.inverse_transform(normal_x_df_cat)
            except:
                x_cat = np.array([[]])
            try:
                x_ord_cont = self.scaler.inverse_transform(normal_x_df_ord_cont)
            except:
                x_ord_cont = np.array([[]])
            x = np.concatenate((x_bin, x_cat, x_ord_cont), axis=1)
        return x
                
def load_dataset(data_str, train_fraction, seed, step):
    """
    Function to load all datasets according to data_str and train_fraction
    """
    if data_str == 'adult':
        binary = ['Sex','NativeCountry','Race']
        categorical = ['WorkClass','MaritalStatus','Occupation','Relationship']
        ordinal = ['EducationLevel','AgeGroup']
        continuous = ['EducationNumber','CapitalGain','CapitalLoss','HoursPerWeek']
        input_cols = binary + categorical + ordinal + continuous
        label = ['label']
        df = pd.read_csv(dataset_dir+'adult/preprocessed_adult.csv', index_col=0)
                
    elif data_str == 'kdd_census':
        binary = ['Sex','Race']
        categorical = ['Industry','Occupation']
        ordinal = []
        continuous = ['Age','WageHour','CapitalGain','CapitalLoss','Dividends','WorkWeeksYear']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Label']
        df = pd.read_csv(dataset_dir+'kdd_census/preprocessed_kdd_census.csv', index_col=0)
    
    elif data_str == 'german':
        binary = ['Sex','Single','Unemployed']
        categorical = ['PurposeOfLoan','InstallmentRate','Housing']
        ordinal = []
        continuous = ['Age','Credit','LoanDuration']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Label']
        df = pd.read_csv(dataset_dir+'german/preprocessed_german.csv', index_col=0)

    elif data_str == 'dutch':
        binary = ['Sex']
        categorical = ['HouseholdPosition','HouseholdSize','Country','EconomicStatus','CurEcoActivity','MaritalStatus']
        ordinal = ['EducationLevel']
        continuous = ['Age']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Occupation']
        df = pd.read_csv(dataset_dir+'dutch/preprocessed_dutch.csv', index_col=0)
    
    elif data_str == 'bank':
        binary = ['Default','Housing','Loan']
        categorical = ['Job','MaritalStatus','Education','Contact','Poutcome']
        ordinal = ['AgeGroup']
        continuous = ['Balance','Day','Duration','Campaign','Pdays','Previous']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Subscribed']
        df = pd.read_csv(dataset_dir+'bank/preprocessed_bank.csv', index_col=0)
        df = df[input_cols + label]

    elif data_str == 'credit':
        binary = ['isMale','isMarried','HasHistoryOfOverduePayments']
        categorical = []
        ordinal = ['TotalOverdueCounts','TotalMonthsOverdue','AgeGroup','EducationLevel']
        continuous = ['MaxBillAmountOverLast6Months','MaxPaymentAmountOverLast6Months','MonthsWithZeroBalanceOverLast6Months',
                'MonthsWithLowSpendingOverLast6Months','MonthsWithHighSpendingOverLast6Months','MostRecentBillAmount',
                'MostRecentPaymentAmount']
        input_cols = binary + categorical + ordinal + continuous
        label = ['NoDefaultNextMonth (label)']
        df = pd.read_csv(dataset_dir+'/credit/preprocessed_credit.csv', index_col=0)

    elif data_str == 'compass':
        # Based on the MACE algorithm Datasets preprocessing (please, see: https://github.com/amirhk/mace)
        binary = ['Race','Sex','ChargeDegree']
        categorical = []
        ordinal = ['PriorsCount','AgeGroup']
        continuous = []
        input_cols = binary + categorical + ordinal + continuous
        label = ['TwoYearRecid (label)']
        df = pd.read_csv(dataset_dir+'/compass/preprocessed_compass.csv', index_col=0)
    
    elif data_str == 'diabetes':
        binary = ['DiabetesMed']
        categorical = ['Race','Sex','A1CResult','Metformin','Chlorpropamide','Glipizide','Rosiglitazone','Acarbose','Miglitol']
        ordinal = ['AgeGroup']
        continuous = ['TimeInHospital','NumProcedures','NumMedications','NumEmergency']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Label']
        df = pd.read_csv(dataset_dir+'/diabetes/preprocessed_diabetes.csv', index_col=0)

    elif data_str == 'student':
        binary = ['School','Sex','AgeGroup','FamilySize','ParentStatus','SchoolSupport','FamilySupport','ExtraPaid','ExtraActivities','Nursery','HigherEdu','Internet','Romantic']
        categorical = ['MotherJob','FatherJob','SchoolReason']
        ordinal = ['MotherEducation','FatherEducation']
        continuous = ['TravelTime','ClassFailures','GoOut']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Grade']
        df = pd.read_csv(dataset_dir+'/student/preprocessed_student.csv', index_col=0)
        df = df[input_cols + label]

    elif data_str == 'oulad':
        binary = ['Sex','Disability']
        categorical = ['Region','CodeModule','CodePresentation','HighestEducation','IMDBand']
        ordinal = ['AgeGroup']
        continuous = ['NumPrevAttempts','StudiedCredits']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Grade']
        df = pd.read_csv(dataset_dir+'/oulad/preprocessed_oulad.csv', index_col=0)

    elif data_str == 'law':
        binary = ['WorkFullTime','Sex','Race']
        categorical = ['FamilyIncome','Tier']
        ordinal = []
        continuous = ['Decile1stYear','Decile3rdYear','LSAT','UndergradGPA','FirstYearGPA','CumulativeGPA']
        input_cols = binary + categorical + ordinal + continuous
        label = ['BarExam']
        df = pd.read_csv(dataset_dir+'/law/preprocessed_law.csv', index_col=0)
        for col in df.columns:
            df[col] = df[col].astype(float)

    elif data_str == 'ionosphere':
        binary = []
        categorical = []
        ordinal = []
        continuous = ['0','2','4','5','6','7','26','30'] # Chosen based on MDI
        input_cols = binary + categorical + ordinal + continuous
        label = ['label']
        df = pd.read_csv(dataset_dir+'/ionosphere/processed_ionosphere.csv', index_col=0)
    
    elif data_str == 'heart':
        binary = ['Sex','BloodSugar']
        categorical = ['ChestPain']
        ordinal = ['ECG']
        continuous = ['Age','RestBloodPressure','Chol']
        input_cols = binary + categorical + ordinal + continuous
        label = ['class']
        df = pd.read_csv(dataset_dir+'heart/preprocessed_heart.csv', index_col=0)

    elif data_str == 'synthetic_athlete':
        binary = ['Sex']
        categorical = ['Diet','Sport','TrainingTime']
        ordinal = []
        continuous = ['Age','SleepHours']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Label']
        df = pd.read_csv(dataset_dir+'synthetic_athlete/preprocessed_synthetic_athlete.csv', index_col=0)
    
    elif data_str == 'synthetic_disease':
        binary = ['Smokes']
        categorical = ['Diet','Stress']
        ordinal = ['Weight']
        continuous = ['Age','ExerciseMinutes','SleepHours']
        input_cols = binary + categorical + ordinal + continuous
        label = ['Label']
        df = pd.read_csv(dataset_dir+'synthetic_disease/preprocessed_synthetic_disease.csv', index_col=0)

    data_obj = Dataset(data_str, seed, train_fraction, label, df,
                   binary, categorical, ordinal, continuous, step)
    return data_obj
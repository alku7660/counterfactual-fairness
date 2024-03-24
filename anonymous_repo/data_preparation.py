from support import dataset_dir
import numpy as np
import pandas as pd
import os

def erase_missing(df):
    """
    DESCRIPTION:        Eliminates instances with missing values
    
    INPUT:
    df:                 The dataset of interest as DataFrame
    
    OUTPUT:
    df:                 Filtered dataset without points with missing values
    """
    df = df.replace({'?':np.nan})
    df = df.replace({' ?':np.nan})
    df.dropna(axis=0,how='any',inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def prepare_data(data_str):
    """
    DESCRIPTION:        Loads and prepares the dataset for preprocessing
                        Preprocessing of some of the files, and some of the files based on or obtained from the MACE algorithm preprocessing (please, see: https://github.com/amirhk/mace)
    INPUT:
    data_str:           Name of the dataset to load

    OUTPUT: (None: Storing data to files)
    """

    if data_str == 'adult':
        binary = ['Sex','NativeCountry','Race']
        categorical = ['WorkClass','MaritalStatus','Occupation','Relationship']
        numerical = ['EducationNumber','CapitalGain','CapitalLoss','HoursPerWeek','EducationLevel','AgeGroup']
        label = ['label']
        attrs = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']  # all attributes
        sensitive_attrs = ['sex']  # the fairness constraints will be used for this feature
        attrs_to_ignore = ['sex','fnlwgt']  # sex and race are sensitive feature so we will not use them in classification, we will not consider fnlwght for classification since its computed externally and it highly predictive for the class (for details, see documentation of the adult data)
        # adult data comes in two different files, one for training and one for testing, however, we will combine data from both the files
        this_files_directory = dataset_dir+data_str+'/'
        data_files = ["adult.data", "adult.test"]
        X = []
        y = []
        x_control = {}
        attrs_to_vals = {}  # will store the values for each attribute for all users
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
                    continue  # skip empty lines
                line = line.split(", ")
                if len(line) != 15 or "?" in line:  # if a line has missing attributes, ignore it
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
                    # reducing dimensionality of some very sparse features
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
    elif data_str == 'german':
        binary = ['Sex','Single','Unemployed']
        categorical = ['PurposeOfLoan','InstallmentRate','Housing']
        numerical = ['Age','Credit','LoanDuration']
        label = ['Label']
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
        processed_df.loc[processed_df['EducationLevel'] == 5,'EducationLevel'] = 6
        processed_df.loc[processed_df['EducationLevel'] == 4,'EducationLevel'] = 5
        processed_df.loc[processed_df['EducationLevel'] == 3,'EducationLevel'] = 4
        processed_df.loc[processed_df['EducationLevel'] == 2,'EducationLevel'] = 3
        processed_df.loc[processed_df['EducationLevel'] == 1,'EducationLevel'] = 2
        processed_df.loc[processed_df['EducationLevel'] == 0,'EducationLevel'] = 1
    elif data_str == 'compass':
        processed_df = pd.DataFrame()
        binary = ['Race','Sex','ChargeDegree']
        categorical = []
        numerical = ['PriorsCount','AgeGroup']
        label = ['TwoYearRecid (label)']
        FEATURES_CLASSIFICATION = ['age_cat','race','sex','priors_count','c_charge_degree']  # features to be used for classification
        CLASS_FEATURE = 'two_year_recid'  # the decision variable
        df = pd.read_csv(dataset_dir+'/compass/compas-scores-two-years.csv')
        df = df.dropna(subset=["days_b_screening_arrest"])  # dropping missing vals
        # """ Data filtering and preparation """ (As seen in MACE algorithm, based on Propublica methodology. Please, see: https://github.com/amirhk/mace)
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
        processed_df.loc[processed_df['PriorsCount'] == 38,'PriorsCount'] = 36
        processed_df.loc[processed_df['PriorsCount'] == 37,'PriorsCount'] = 35
        processed_df.loc[processed_df['PriorsCount'] == 36,'PriorsCount'] = 34
        processed_df.loc[processed_df['PriorsCount'] == 33,'PriorsCount'] = 33
        processed_df.loc[processed_df['PriorsCount'] == 31,'PriorsCount'] = 32
        processed_df.loc[processed_df['PriorsCount'] == 30,'PriorsCount'] = 31
        processed_df.loc[processed_df['PriorsCount'] == 29,'PriorsCount'] = 30
        processed_df.loc[processed_df['PriorsCount'] == 28,'PriorsCount'] = 29
        processed_df.loc[processed_df['PriorsCount'] == 27,'PriorsCount'] = 28
        processed_df.loc[processed_df['PriorsCount'] == 26,'PriorsCount'] = 27
        processed_df.loc[processed_df['PriorsCount'] == 25,'PriorsCount'] = 26
        processed_df.loc[processed_df['PriorsCount'] == 24,'PriorsCount'] = 25
        processed_df.loc[processed_df['PriorsCount'] == 23,'PriorsCount'] = 24
        processed_df.loc[processed_df['PriorsCount'] == 22,'PriorsCount'] = 23
        processed_df.loc[processed_df['PriorsCount'] == 21,'PriorsCount'] = 22
        processed_df.loc[processed_df['PriorsCount'] == 20,'PriorsCount'] = 21
        processed_df.loc[processed_df['PriorsCount'] == 19,'PriorsCount'] = 20
        processed_df.loc[processed_df['PriorsCount'] == 18,'PriorsCount'] = 19
        processed_df.loc[processed_df['PriorsCount'] == 17,'PriorsCount'] = 18
        processed_df.loc[processed_df['PriorsCount'] == 16,'PriorsCount'] = 17
        processed_df.loc[processed_df['PriorsCount'] == 15,'PriorsCount'] = 16
        processed_df.loc[processed_df['PriorsCount'] == 14,'PriorsCount'] = 15
        processed_df.loc[processed_df['PriorsCount'] == 13,'PriorsCount'] = 14
        processed_df.loc[processed_df['PriorsCount'] == 12,'PriorsCount'] = 13
        processed_df.loc[processed_df['PriorsCount'] == 11,'PriorsCount'] = 12
        processed_df.loc[processed_df['PriorsCount'] == 10,'PriorsCount'] = 11
        processed_df.loc[processed_df['PriorsCount'] == 9,'PriorsCount'] = 10
        processed_df.loc[processed_df['PriorsCount'] == 8,'PriorsCount'] = 9
        processed_df.loc[processed_df['PriorsCount'] == 7,'PriorsCount'] = 8
        processed_df.loc[processed_df['PriorsCount'] == 6,'PriorsCount'] = 7
        processed_df.loc[processed_df['PriorsCount'] == 5,'PriorsCount'] = 6
        processed_df.loc[processed_df['PriorsCount'] == 4,'PriorsCount'] = 5
        processed_df.loc[processed_df['PriorsCount'] == 3,'PriorsCount'] = 4
        processed_df.loc[processed_df['PriorsCount'] == 2,'PriorsCount'] = 3
        processed_df.loc[processed_df['PriorsCount'] == 1,'PriorsCount'] = 2
        processed_df.loc[processed_df['PriorsCount'] == 0,'PriorsCount'] = 1
        processed_df.loc[df['c_charge_degree'] == 'M', 'ChargeDegree'] = 1
        processed_df.loc[df['c_charge_degree'] == 'F', 'ChargeDegree'] = 2
        processed_df = processed_df.reset_index(drop=True)
    elif data_str == 'student':
        binary = ['School','Sex','AgeGroup','Address','FamilySize','ParentStatus','SchoolSupport','FamilySupport','ExtraPaid','ExtraActivities','Nursery','HigherEdu','Internet','Romantic']
        categorical = ['MotherJob','FatherJob','SchoolReason']
        numerical = ['MotherEducation','FatherEducation','TravelTime','ClassFailures','GoOut']
        label = ['Grade']
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
    processed_df.to_csv(f'{dataset_dir}/{data_str}/preprocessed_{data_str}.csv')

data_str = 'adult' # or any dataset found here
prepare_data(data_str)
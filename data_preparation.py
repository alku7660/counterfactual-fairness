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
    elif data_str == 'kdd_census':
        binary = ['Sex','Race']
        categorical = ['Industry','Occupation']
        numerical = ['Age','WageHour','CapitalGain','CapitalLoss','Dividends','WorkWeeksYear']
        label = ['Label']
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
    elif data_str == 'bank':
        binary = ['Default','Housing','Loan']
        categorical = ['Job','MaritalStatus','Education','Contact','Month','Poutcome']
        numerical = ['AgeGroup','Balance','Day','Duration','Campaign','Pdays','Previous']
        label = ['Subscribed']
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
        processed_df.loc[df['c_charge_degree'] == 'M', 'ChargeDegree'] = 1
        processed_df.loc[df['c_charge_degree'] == 'F', 'ChargeDegree'] = 2
        processed_df = processed_df.reset_index(drop=True)
    elif data_str == 'diabetes':
        binary = ['DiabetesMed']
        categorical = ['Race','Sex','A1CResult','Metformin','Chlorpropamide','Glipizide','Rosiglitazone','Acarbose','Miglitol']
        numerical = ['AgeGroup','TimeInHospital','NumProcedures','NumMedications','NumEmergency']
        label = ['Label']
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
        processed_df.reset_index(drop=True, inplace=True)
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
    elif data_str == 'oulad':
        binary = ['Sex','Disability']
        categorical = ['Region','CodeModule','CodePresentation','HighestEducation','IMDBand']
        numerical = ['NumPrevAttempts','StudiedCredits','AgeGroup']
        label = ['Grade']
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
    
    processed_df.to_csv(f'{dataset_dir}/{data_str}/preprocessed_{data_str}.csv')

data_str = 'adult'
prepare_data(data_str)
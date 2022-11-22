import copy
import pandas as pd

def define_feat_type(data, framework='normal'):
    """
    DESCRIPTION:        Obtains a feature type vector corresponding to each of the features
    
    INPUT:
    data:               Data object

    OUTPUT:
    feat_type:          Dataset feature type series
    """
    if framework == 'carla':
        feat_type = data.carla_transformed_train_df.dtypes
    else:
        feat_type = data.transformed_train_df.dtypes
    feat_type_out = copy.deepcopy(feat_type)
    feat_list = feat_type.index.tolist()
    if data.name == 'adult':
        for i in feat_list:
            if framework == 'carla':
                if 'Sex' in i or 'Native' in i or 'WorkClass' in i or 'Marital' in i or 'Occupation' in i or 'Relation' in i or 'Race' in i or 'Age' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'EducationNumber' in i or 'Capital' in i or 'Hours' in i or 'EducationLevel' in i:
                    feat_type_out.loc[i] = 'num-con'
            else:
                if 'Sex' in i or 'Native' in i or 'WorkClass' in i or 'Marital' in i or 'Occupation' in i or 'Relation' in i or 'Race' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'EducationLevel' in i or 'Age' in i:
                    feat_type_out.loc[i] = 'num-ord'
                elif 'EducationNumber' in i or 'Capital' in i or 'Hours' in i:
                    feat_type_out.loc[i] = 'num-con'
    elif data.name == 'kdd_census':
        for i in feat_list:
            if framework == 'carla':
                if 'Sex' in i or 'Race' in i or 'Industry' in i or 'Occupation' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Age' in i or 'WageHour' in i or 'CapitalGain' in i or 'CapitalLoss' in i or 'Dividends' in i or 'WorkWeeksYear' in i:
                    feat_type_out.loc[i] = 'num-con'
            else:
                if 'Sex' in i or 'Race' in i or 'Industry' in i or 'Occupation' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Age' in i or 'WageHour' in i or 'CapitalGain' in i or 'CapitalLoss' in i or 'Dividends' in i or 'WorkWeeksYear' in i:
                    feat_type_out.loc[i] = 'num-con'
    elif data.name == 'german':
        for i in feat_list:
            if framework == 'carla':
                if 'Sex' in i or 'Single' in i or 'Unemployed' in i or 'Housing' in i or 'PurposeOfLoan' in i or 'InstallmentRate' in i or 'Housing' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Age' in i or 'Credit' in i or 'Loan' in i:
                    feat_type_out.loc[i] = 'num-con'
            else:
                if 'Sex' in i or 'Single' in i or 'Unemployed' in i or 'Housing' in i or 'PurposeOfLoan' in i or 'InstallmentRate' in i or 'Housing' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Age' in i or 'Credit' in i or 'Loan' in i:
                    feat_type_out.loc[i] = 'num-con'
    elif data.name == 'dutch':
        for i in feat_list:
            if framework == 'carla':
                if 'Sex' in i or 'HouseholdPosition' in i or 'HouseholdSize' in i or 'Country' in i or 'EconomicStatus' in i or 'CurEcoActivity' in i or 'MaritalStatus' in i or 'EducationLevel' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Age' in i:
                    feat_type_out.loc[i] = 'num-con'
            else:
                if 'Sex' in i or 'HouseholdPosition' in i or 'HouseholdSize' in i or 'Country' in i or 'EconomicStatus' in i or 'CurEcoActivity' in i or 'MaritalStatus' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'EducationLevel' in i:
                    feat_type_out.loc[i] = 'num-ord'
                elif 'Age' in i:
                    feat_type_out.loc[i] = 'num-con'
    elif data.name == 'bank':
        for i in feat_list:
            if framework == 'carla':
                if 'Default' in i or 'Housing' in i or 'Loan' in i or 'Job' in i or 'MaritalStatus' in i or 'Education' in i or 'Contact' in i or 'Month' in i or 'Poutcome' in i or 'Age' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Balance' in i or 'Day' in i or 'Duration' in i or 'Campaign' in i or 'Pdays' in i or 'Previous' in i:
                    feat_type_out.loc[i] = 'num-con'
            else:    
                if 'Default' in i or 'Housing' in i or 'Loan' in i or 'Job' in i or 'MaritalStatus' in i or 'Education' in i or 'Contact' in i or 'Month' in i or 'Poutcome' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Age' in i:
                    feat_type_out.loc[i] = 'num-ord'
                elif 'Balance' in i or 'Day' in i or 'Duration' in i or 'Campaign' in i or 'Pdays' in i or 'Previous' in i:
                    feat_type_out.loc[i] = 'num-con'
    elif data.name == 'credit':
        for i in feat_list:
            if framework == 'carla':
                if 'Male' in i or 'Married' in i or 'History' in i or 'Age' in i or 'Education' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Amount' in i or 'Balance' in i or 'Spending' in i or 'Total' in i:
                    feat_type_out.loc[i] = 'num-con'
            else:
                if 'Male' in i or 'Married' in i or 'History' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Amount' in i or 'Balance' in i or 'Spending' in i:
                    feat_type_out.loc[i] = 'num-con'
                elif 'Total' in i:
                    feat_type_out.loc[i] = 'num-ord'
    elif data.name == 'compass':
        for i in feat_list:
            if framework == 'carla':
                if 'Sex' in i or 'Race' in i or 'Charge' in i or 'Age' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Priors' in i:
                    feat_type_out.loc[i] = 'num-con'
            else:
                if 'Sex' in i or 'Race' in i or 'Charge' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Priors' in i or 'Age' in i:
                    feat_type_out.loc[i] = 'num-ord'
    elif data.name == 'diabetes':
        for i in feat_list:
            if framework == 'carla':
                if 'DiabetesMed' in i or 'Race' in i or 'Sex' in i or 'A1CResult' in i or 'Metformin' in i or 'Chlorpropamide' in i or 'Glipizide' in i or 'Rosiglitazone' in i or 'Acarbose' in i or 'Miglitol' in i or 'AgeGroup' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'TimeInHospital' in i or 'NumProcedures' in i or 'NumMedications' in i or 'NumEmergency':
                    feat_type_out.loc[i] = 'num-con'
            else:
                if 'DiabetesMed' in i or 'Race' in i or 'Sex' in i or 'A1CResult' in i or 'Metformin' in i or 'Chlorpropamide' in i or 'Glipizide' in i or 'Rosiglitazone' in i or 'Acarbose' in i or 'Miglitol' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'AgeGroup' in i:
                    feat_type_out.loc[i] = 'num-ord'
                elif 'TimeInHospital' in i or 'NumProcedures' in i or 'NumMedications' in i or 'NumEmergency':
                    feat_type_out.loc[i] = 'num-con'
    elif data.name == 'student':
        for i in feat_list:
            if framework == 'carla':
                if 'Age' in i or 'School' in i or 'Sex' in i or 'Address' in i or 'FamilySize' in i or 'ParentStatus' in i or 'SchoolSupport' in i or 'FamilySupport' in i or 'ExtraPaid' in i or 'ExtraActivities' in i or 'Nursery' in i or 'HigherEdu' in i or 'Internet' in i or 'Romantic' in i or 'MotherJob' in i or 'FatherJob' in i or 'SchoolReason' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'TravelTime' in i or 'ClassFailures' in i or 'GoOut' in i or 'MotherEducation' in i or 'FatherEducation' in i:
                    feat_type_out.loc[i] = 'num-con'
            else:
                if 'Age' in i or 'School' in i or 'Sex' in i or 'Address' in i or 'FamilySize' in i or 'ParentStatus' in i or 'SchoolSupport' in i or 'FamilySupport' in i or 'ExtraPaid' in i or 'ExtraActivities' in i or 'Nursery' in i or 'HigherEdu' in i or 'Internet' in i or 'Romantic' in i or 'MotherJob' in i or 'FatherJob' in i or 'SchoolReason' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'MotherEducation' in i or 'FatherEducation' in i:
                    feat_type_out.loc[i] = 'num-ord'
                elif 'TravelTime' in i or 'ClassFailures' in i or 'GoOut' in i:
                    feat_type_out.loc[i] = 'num-con'
    elif data.name == 'oulad':
        for i in feat_list:
            if framework == 'carla':
                if 'Sex' in i or 'Disability' in i or 'Region' in i or 'CodeModule' in i or 'CodePresentation' in i or 'HighestEducation' in i or 'IMDBand' in i or 'AgeGroup' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'NumPrevAttempts' in i or 'StudiedCredits' in i:
                    feat_type_out.loc[i] = 'num-con'
            else:
                if 'Sex' in i or 'Disability' in i or 'Region' in i or 'CodeModule' in i or 'CodePresentation' in i or 'HighestEducation' in i or 'IMDBand' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'NumPrevAttempts' in i or 'StudiedCredits' in i:
                    feat_type_out.loc[i] = 'num-con'
                elif 'AgeGroup' in i:
                    feat_type_out.loc[i] = 'num-ord'
    elif data.name == 'law':
        for i in feat_list:
            if framework == 'carla':
                if 'FamilyIncome' in i or 'Tier' in i or 'Race' in i or 'WorkFullTime' in i or 'Sex' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Decile1stYear' in i or 'Decile3rdYear' in i or 'LSAT' in i or 'UndergradGPA' in i or 'FirstYearGPA' in i or 'CumulativeGPA' in i:
                    feat_type_out.loc[i] = 'num-con'
            else:
                if 'FamilyIncome' in i or 'Tier' in i or 'Race' in i or 'WorkFullTime' in i or 'Sex' in i:
                    feat_type_out.loc[i] = 'bin'
                elif 'Decile1stYear' in i or 'Decile3rdYear' in i or 'LSAT' in i or 'UndergradGPA' in i or 'FirstYearGPA' in i or 'CumulativeGPA' in i:
                    feat_type_out.loc[i] = 'num-con'
    return feat_type_out

def define_protected(data):
    """
    DESCRIPTION:        Defines which features are sensitive / protected and the groups or categories in each of them
    
    INPUT:
    data:               Data object

    OUTPUT:
    feat_protected:     Protected set of features per dataset
    """
    feat_protected = dict()
    if data.name == 'adult':
        feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
        feat_protected['Race'] = {1.00:'White', 2.00:'Non-white'}
        feat_protected['AgeGroup'] = {1.00:'<25', 2.00:'25-60', 3.00:'>60'}
    elif data.name == 'kdd_census':
        feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
        feat_protected['Race'] = {1.00:'White', 2.00:'Non-white'}
    elif data.name == 'german':
        feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
    elif data.name == 'dutch':
        feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
    elif data.name == 'bank':
        feat_protected['AgeGroup'] = {1.00:'<25', 2.00:'25-60', 3.00:'>60'}
        feat_protected['MaritalStatus'] = {1.00:'Married', 2.00:'Single', 3.00:'Divorced'}
    elif data.name == 'credit':
        feat_protected['isMale'] = {1.00:'True', 0.00:'False'}
        feat_protected['isMarried'] = {1.00:'True', 0.00:'False'}
        feat_protected['EducationLevel'] = {1.00:'Other', 2.00:'HS', 3.00:'University', 4.00:'Graduate'}
    elif data.name == 'compass':
        feat_protected['Race'] = {1.00:'African-American', 2.00:'Caucasian'}
        feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
    elif data.name == 'diabetes':
        feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
    elif data.name == 'student':
        feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
        feat_protected['AgeGroup'] = {1.00:'<18', 2.00:'>=18'}
    elif data.name == 'oulad':
        feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
    elif data.name == 'law':
        feat_protected['Sex'] = {1.00:'Male', 2.00:'Female'}
        feat_protected['Race'] = {1.00:'White', 2.00:'Non-white'}
    return feat_protected

def define_mutability(data, feat_protected, framework='normal'):
    """
    DESCRIPTION:        Method that outputs mutable features per dataset

    INPUT:
    data:               Data object
    feat_protected:     Dictionary contatining the protected features and the sensitive groups names

    OUTPUT:
    feat_mutable:       Series indicating the mutability of each feature
    """
    if framework == 'carla':
        feat_list = data.carla_transformed_cols
    else:
        feat_list = data.transformed_cols
    feat_mutable  = dict()
    for i in feat_list:
        feat_mutable[i] = 1
    for i in feat_protected.keys():
        idx_feat_protected = [j for j in range(len(feat_list)) if i in feat_list[j]]
        feat = [feat_list[j] for j in idx_feat_protected]
        for j in feat:
            feat_mutable[j] = 0
    feat_mutable = pd.Series(feat_mutable)
    return feat_mutable
    
def define_directionality(data, framework='normal'):
    """
    DESCRIPTION:        Method that outputs change directionality of features per dataset
    
    INPUT:
    data:               Data object

    OUTPUT:
    feat_dir:           Series containing plausible direction of change of each feature
    """
    if framework == 'carla':
        feat_list = data.carla_transformed_cols
    else:
        feat_list = data.transformed_cols
    feat_dir  = dict()
    if data.name == 'adult':
        for i in feat_list:
            if 'Age' in i or 'Sex' in i or 'Race' in i:
                feat_dir[i] = 0
            elif 'Education' in i:
                feat_dir[i] = 'pos'
            else:
                feat_dir[i] = 'any'
    elif data.name == 'kdd_census':
        for i in feat_list:
            if 'Sex' in i or 'Race' in i:
                feat_dir[i] = 0
            elif 'Industry' in i or 'Occupation' in i or 'WageHour' in i or 'CapitalGain' in i or 'CapitalLoss' in i or 'Dividends' in i or 'WorkWeeksYear' or 'Age' in i:
                feat_dir[i] = 'any'
    elif data.name == 'german':
        for i in feat_list:
            if 'Age' in i or 'Sex' in i:
                feat_dir[i] = 0
            else:
                feat_dir[i] = 'any'
    elif data.name == 'dutch':
        for i in feat_list:
            if 'Sex' in i:
                feat_dir[i] = 0
            elif 'HouseholdPosition' in i or 'HouseholdSize' in i or 'EconomicStatus' in i or 'CurEcoActivity' in i or 'MaritalStatus' in i or 'Country' in i:
                feat_dir[i] = 'any'
            elif 'EducationLevel' in i or 'Age' in i:
                feat_dir[i] = 'pos'
    elif data.name == 'bank':
        for i in feat_list:
            if 'Age' in i or 'Marital' in i:
                feat_dir[i] = 0
            elif 'Default' in i or 'Housing' in i or 'Loan' in i or 'Job' in i or 'Contact' in i or 'Month' in i or 'Poutcome' or 'Balance' in i or 'Day' in i or 'Duration' in i or 'Campaign' in i or 'Pdays' in i or 'Previous' in i:
                feat_dir[i] = 'any'
            elif 'Education' in i:
                feat_dir[i] = 'pos'
    elif data.name == 'credit':
        for i in feat_list:
            if 'Age' in i or 'Male' in i:
                feat_dir[i] = 0
            elif 'OverLast6Months' in i or 'MostRecent' in i or 'Total' in i or 'History' in i or 'Married' in i:
                feat_dir[i] = 'any'   
            elif 'Education' in i:
                feat_dir[i] = 'pos'
    elif data.name == 'compass':
        for i in feat_list:
            if 'Age' in i or 'Sex' in i or 'Race' in i:
                feat_dir[i] = 0
            elif 'Charge' in i or 'Priors' in i:
                feat_dir[i] = 'any'
    elif data.name == 'diabetes':
        for i in feat_list:
            if 'Sex' in i:
                feat_dir[i] = 0
            else:
                feat_dir[i] = 'any'
    elif data.name == 'student':
        for i in feat_list:
            if 'Sex' in i or 'Age' in i:
                feat_dir[i] = 0
            else:
                feat_dir[i] = 'any'
    elif data.name == 'oulad':
        for i in feat_list:
            if 'Sex' in i:
                feat_dir[i] = 0
            else:
                feat_dir[i] = 'any'
    elif data.name == 'law':
        for i in feat_list:
            if 'Sex' in i or 'Race' in i:
                feat_dir[i] = 0
            else:
                feat_dir[i] = 'any'
    feat_dir = pd.Series(feat_dir)
    return feat_dir

def define_feat_cost(data, framework='normal'):
    """
    DESCRIPTION:        Allocates a unit cost of change to the features of the datasets

    INPUT:
    data:               Dataset object

    OUTPUT:
    feat_cost:          Series with the theoretical unit cost of changing each feature
    """
    if framework == 'carla':
        feat_list = data.carla_transformed_cols
    else:
        feat_list = data.transformed_cols
    feat_cost  = dict()
    if data.name == 'adult':
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
    elif data.name == 'kdd_census':
        for i in feat_list:
            if 'Sex' in i or 'Race' in i:
                feat_cost[i] = 0
            elif 'Industry' in i or 'Occupation' in i or 'WageHour' in i or 'CapitalGain' in i or 'CapitalLoss' in i or 'Dividends' in i or 'WorkWeeksYear':
                feat_cost[i] = 1
    elif data.name == 'german':
        for i in feat_list:
            if 'Age' in i or 'Sex' in i:
                feat_cost[i] = 0
            else:
                feat_cost[i] = 1
    elif data.name == 'dutch':
        for i in feat_list:
            if 'Sex' in i:
                feat_cost[i] = 0
            elif 'HouseholdPosition' in i or 'HouseholdSize' in i or 'EconomicStatus' in i or 'CurEcoActivity' in i or 'MaritalStatus' in i or 'EducationLevel' in i or 'Age' in i or 'Country' in i:
                feat_cost[i] = 1
    elif data.name == 'bank':
        for i in feat_list:
            if 'Age' in i or 'Marital' in i:
                feat_cost[i] = 0
            else:
                feat_cost[i] = 1
    elif data.name == 'credit':
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
    elif data.name == 'compass':
        for i in feat_list:
            if 'Age' in i or 'Sex' in i or 'Race' in i:
                feat_cost[i] = 0
            elif 'Charge' in i:
                feat_cost[i] = 1#10
            elif 'Priors' in i:
                feat_cost[i] = 1#20
    elif data.name == 'diabetes':
        for i in feat_list:
            if 'Sex' in i:
                feat_cost[i] = 0
            else:
                feat_cost[i] = 1
    elif data.name == 'student':
        for i in feat_list:
            if 'Sex' in i or 'Age' in i:
                feat_cost[i] = 0
            else:
                feat_cost[i] = 1
    elif data.name == 'oulad':
        for i in feat_list:
            if 'Sex' in i:
                feat_cost[i] = 0
            else:
                feat_cost[i] = 1
    elif data.name == 'law':
        for i in feat_list:
            if 'Sex' in i or 'Race' in i:
                feat_cost[i] = 0
            else:
                feat_cost[i] = 1
    feat_cost = pd.Series(feat_cost)
    return feat_cost

def define_feat_step(data, feat_type, framework='normal'):
    """
    DESCRIPTION:        Estimates the step size of all features (used for ordinal features)

    INPUT:
    data:               Dataset object

    OUTPUT:
    feat_step:          Plausible step size for each feature 
    """
    if framework == 'carla':
        scaler = data.carla_scaler
    else:
        scaler = data.scaler
    feat_step = pd.Series(data=1/(scaler.data_max_ - scaler.data_min_), index=[i for i in feat_type.keys() if feat_type[i] in ['num-ord','num-con']])
    for i in feat_type.keys().tolist():
        if feat_type.loc[i] == 'num-con':
            feat_step.loc[i] = data.step
        elif feat_type.loc[i] == 'num-ord':
            continue
        else:
            feat_step.loc[i] = 0
    feat_step = feat_step.reindex(index = feat_type.keys().to_list())
    return feat_step

def define_category_groups(data, feat_type):
    """
    DESCRIPTION:        Method that assigns categorical groups to different one-hot encoded categorical features
    
    INPUT:
    data:               Dataset object

    OUTPUT:
    feat_cat:           Category groups for each of the features
    """
    feat_cat = copy.deepcopy(feat_type)
    feat_list = feat_type.index.tolist()
    if data.name == 'adult':
        for i in feat_list:
            if 'Sex' in i or 'Native' in i or 'EducationLevel' in i or 'EducationNumber' in i or 'Capital' in i or 'Hours' in i or 'Race' in i:
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
    elif data.name == 'kdd_census':
        for i in feat_list:
            if 'Industry' in i:
                feat_cat.loc[i] = 'cat_0'
            elif 'Occupation' in i:
                feat_cat.loc[i] = 'cat_1'    
            else:
                feat_cat.loc[i] = 'non'
    elif data.name == 'german':
        for i in feat_list:
            if 'PurposeOfLoan' in i:
                feat_cat.loc[i] = 'cat_0'
            elif 'InstallmentRate' in i:
                feat_cat.loc[i] = 'cat_1'
            elif 'Housing' in i:
                feat_cat.loc[i] = 'cat_2'
            else:
                feat_cat.loc[i] = 'non'
    elif data.name == 'dutch':
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
    elif data.name == 'bank':
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
    elif data.name == 'credit':
        for i in feat_list:
            feat_cat.loc[i] = 'non'
    elif data.name == 'compass':
        for i in feat_list:
            feat_cat.loc[i] = 'non'
    elif data.name == 'diabetes':
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
    elif data.name == 'student':
        for i in feat_list:
            if 'MotherJob' in i:
                feat_cat.loc[i] = 'cat_0'
            elif 'FatherJob' in i:
                feat_cat.loc[i] = 'cat_1'
            elif 'SchoolReason' in i:
                feat_cat.loc[i] = 'cat_2'
            else:
                feat_cat.loc[i] = 'non'
    elif data.name == 'oulad':
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
    elif data.name == 'law':
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

def define_all_parameters(data):
    """
    DESCRIPTION:        Returns all relevant dataset parameters

    INPUT:
    data:               Dataset object

    OUTPUT:
    feat_type:          Series containing the feature types ('bin' for binary and categorical, 'num-ord' for ordinal, 'num-con' for continuous)
    feat_protected:     Dictionary containing the protected or sensitive groups
    feat_mutable:       Series indicating the mutability of each feature
    feat_dir:           Series containing plausible direction of change of each feature
    feat_cost:          Series with the theoretical unit cost of changing each feature
    feat_step:          Plausible step size for each feature
    feat_cat:           Category groups for each of the features
    """
    feat_type = define_feat_type(data)
    feat_protected = define_protected(data)
    feat_mutable = define_mutability(data, feat_protected)
    feat_dir = define_directionality(data)
    feat_cost = define_feat_cost(data)
    feat_step = define_feat_step(data, feat_type)
    feat_cat = define_category_groups(data, feat_type)
    return feat_type, feat_protected, feat_mutable, feat_dir, feat_cost, feat_step, feat_cat

def carla_define_all_parameters(data):
    """
    DESCRIPTION:        Returns all relevant dataset parameters

    INPUT:
    data:               Dataset object

    OUTPUT:
    feat_type:          Series containing the feature types ('bin' for binary and categorical, 'num-ord' for ordinal, 'num-con' for continuous)
    feat_protected:     Dictionary containing the protected or sensitive groups
    feat_mutable:       Series indicating the mutability of each feature
    feat_dir:           Series containing plausible direction of change of each feature
    feat_cost:          Series with the theoretical unit cost of changing each feature
    feat_step:          Plausible step size for each feature
    feat_cat:           Category groups for each of the features
    """
    carla_feat_type = define_feat_type(data, framework='carla')
    carla_feat_mutable = define_mutability(data, data.feat_protected, framework='carla')
    carla_feat_dir = define_directionality(data, framework='carla')
    carla_feat_cost = define_feat_cost(data, framework='carla')
    carla_feat_step = define_feat_step(data, carla_feat_type, framework='carla')
    carla_feat_cat = define_category_groups(data, carla_feat_type)
    return carla_feat_type, carla_feat_mutable, carla_feat_dir, carla_feat_cost, carla_feat_step, carla_feat_cat
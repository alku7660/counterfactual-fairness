from carla import Data
from carla import MLModel

 # Custom data set implementations need to inherit from the Data interface
class MyOwnDataSet(Data):
    def __init__(self,data_obj):
        # The data set can e.g. be loaded in the constructor
        self._dataset = data_obj.train_pd
        self._name = data_obj.name
        self._categorical = data_obj.carla_categorical
        self._continuous = data_obj.carla_continuous

    # List of all categorical features
    @property
    def categoricals(self):
        return self._categorical

    # List of all continous features
    @property
    def continous(self):
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
    def raw(self):
        return self._dataset

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
        self.encoder = data.oh_carla_enc
        self.features = data.carla_trained_features

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
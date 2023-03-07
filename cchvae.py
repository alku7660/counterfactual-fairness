"""
C-CHVAE
Based on:
    (1) original authors implementation: Please see https://github.com/MartinPawel/c-chvae
    (2) CARLA framework: Please see https://github.com/carla-recourse/CARLA
"""

"""
Imports
"""
from abc import ABC, abstractmethod
from address import results_obj
from address import dataset_dir
from typing import Union
import numpy as np
from numpy import linalg as LA
import pandas as pd
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import time

class RecourseMethod(ABC):
    """
    Abstract class to implement custom recourse methods for a given black-box-model.
    """
    def __init__(self, mlmodel):
        self._mlmodel = mlmodel

    @abstractmethod
    def get_counterfactuals(self, factuals: pd.DataFrame):
        """
        Generate counterfactual examples for given factuals.
        """
        pass

class Data(ABC):
    """
    Abstract class to implement arbitrary datasets, which are provided by the user.
    """
    @property
    @abstractmethod
    def categorical(self):
        """
        Provides the column names of categorical data.
        Column names do not contain encoded information as provided by a get_dummy() method (e.g., sex_female)
        """
        pass

    @property
    @abstractmethod
    def continuous(self):
        """
        Provides the column names of continuous data.
        """
        pass

    @property
    @abstractmethod
    def immutables(self):
        """
        Provides the column names of immutable data.
        """
        pass

    @property
    @abstractmethod
    def target(self):
        """
        Provides the name of the label column.
        """
        pass

    @property
    @abstractmethod
    def df(self):
        """
        The full Dataframe.
        """
        pass

    @property
    @abstractmethod
    def df_train(self):
        """
        The training split Dataframe.
        """
        pass

    @property
    @abstractmethod
    def df_test(self):
        """
        The testing split Dataframe.
        """
        pass

    @abstractmethod
    def transform(self, df):
        """
        Data transformation, for example normalization of continuous features and encoding of categorical features.
        """
        pass

    @abstractmethod
    def inverse_transform(self, df):
        """
        Inverts transform operation.
        """
        pass

class MLModel(ABC):
    """
    Abstract class to implement custom black-box-model for a given dataset with encoding and scaling processing.
    """

    def __init__(self, data: Data):
        self._data: Data = data

    @property
    def data(self) -> Data:
        """
        Contains the data.api.Data dataset.
        """
        return self._data

    @data.setter
    def data(self, data: Data) -> None:
        self._data = data

    @property
    @abstractmethod
    def feature_input_order(self):
        """
        Saves the required order of features as list.
        """
        pass

    @property
    @abstractmethod
    def backend(self):
        """
        Describes the type of backend which is used for the classifier.
        """
        pass

    @property
    @abstractmethod
    def raw_model(self):
        """
        Contains the raw ML model built on its framework
        """
        pass

    @abstractmethod
    def predict(self, x: Union[np.ndarray, pd.DataFrame]):
        """
        One-dimensional prediction of ml model for an output interval of [0, 1].
        """
        pass

    @abstractmethod
    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame]):
        """
        Two-dimensional probability prediction of ml model.
        """
        pass

    def get_ordered_features(self, x):
        """
        Restores the correct input feature order for the ML model, this also drops the target column.
        """
        if isinstance(x, pd.DataFrame):
            return order_data(self.feature_input_order, x)
        else:
            warnings.warn(f"cannot re-order features for non dataframe input: {type(x)}")
            return x

class MyOwnDataSet(Data):
    def __init__(self, data_obj):
        self._dataset_train = data_obj.transformed_train_df
        self._dataset_test = data_obj.transformed_test_df
        self._dataset = pd.concat([self._dataset_train, self._dataset_test], axis=0)
        self._dataset[data_obj.label_name] = pd.DataFrame(np.concatenate((data_obj.train_target, data_obj.test_target)), index=list(self._dataset.index))
        self._name = data_obj.name
        self._categorical = data_obj.binary + data_obj.categorical
        self._continuous = data_obj.ordinal + data_obj.continuous
        self.encoder = data_obj.bin_cat_enc
        self.scaler = data_obj.scaler
        self._enc_categorical = data_obj.bin_cat_enc_cols
        self.processed_features = self.continuous + self._enc_categorical
        self.undesired_class = data_obj.undesired_class
    
    @property
    def categorical(self):
        return self._categorical

    @property
    def continuous(self):
        return self._continuous
    
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
        elif self._name == 'synthetic_athlete' or self._name == 'synthetic_disease':
            label = 'Label'
        elif self._name == 'heart':
            label = 'class'
        return label
    
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

class MyOwnModel(MLModel):
    def __init__(self, data, model):
        super().__init__(data)
        self._mymodel = model.model
        self.undesired_class = data.undesired_class
        self.scaler = data.scaler
        self.encoder = data.encoder
        self.features = data.processed_features

    @property
    def feature_input_order(self):
        return self.features

    @property
    def backend(self):
        return "sklearn"

    @property
    def raw_model(self):
        return self._mymodel

    def predict(self, x):
        return self._mymodel.predict(x)

    def predict_proba(self, x):
        return self._mymodel.predict_proba(x)

class VAEDataset(Dataset):
    """
    Reads dataframe where last column is the label and the other columns are the features.
    """
    def __init__(self, data: np.ndarray, with_target=True):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if with_target:
            self.X_train = torch.tensor(data[:, :-1], dtype=torch.float32).to(device)
            self.Y_train = torch.tensor(data[:, -1:]).to(device)
        else:
            self.X_train = torch.tensor(data, dtype=torch.float32).to(device)
        self.with_target = with_target

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        if self.with_target:
            return self.X_train[idx], self.Y_train[idx]
        else:
            return self.X_train[idx]

class VariationalAutoencoder(nn.Module): 
    def __init__(self, data_name: str, layers):
        super(VariationalAutoencoder, self).__init__()

        if len(layers) < 2:
            raise ValueError("Number of layers have to be at least 2 (input and latent space), and number of neurons bigger than 0")

        self._data_name = data_name
        self._input_dim = layers[0]
        latent_dim = layers[-1]

        # The VAE components
        lst_encoder = []
        for i in range(1, len(layers) - 1):
            lst_encoder.append(nn.Linear(layers[i - 1], layers[i]))
            lst_encoder.append(nn.BatchNorm1d(layers[i]))
            lst_encoder.append(nn.ReLU())
        encoder = nn.Sequential(*lst_encoder)

        self._mu_enc = nn.Sequential(encoder, nn.Linear(layers[-2], latent_dim))
        self._log_var_enc = nn.Sequential(encoder, nn.Linear(layers[-2], latent_dim))

        lst_decoder = []
        for i in range(len(layers) - 2, 0, -1):
            lst_decoder.append(nn.Linear(layers[i + 1], layers[i]))
            lst_decoder.append(nn.BatchNorm1d(layers[i]))
            lst_decoder.append((nn.ReLU()))
        decoder = nn.Sequential(*lst_decoder)

        self.mu_dec = nn.Sequential(decoder, nn.Linear(layers[1], self._input_dim), nn.BatchNorm1d(self._input_dim), nn.Sigmoid())
        self.log_var_dec = nn.Sequential(decoder, nn.Linear(layers[1], self._input_dim), nn.BatchNorm1d(self._input_dim), nn.Sigmoid())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

    def encode(self, x):
        return self._mu_enc(x), self._log_var_enc(x)

    def decode(self, z):
        return self.mu_dec(z), self.log_var_dec(z)

    def __reparametrization_trick(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # the Gaussian random noise
        return mu + std * epsilon

    def forward(self, x):
        mu_z, log_var_z = self.encode(x)
        z_rep = self.__reparametrization_trick(mu_z, log_var_z)
        mu_x, log_var_x = self.decode(z_rep)
        return mu_x, log_var_x, z_rep, mu_z, log_var_z

    def predict(self, data):
        return self.forward(data)

    def regenerate(self, z):
        mu_x, log_var_x = self.decode(z)
        return mu_x

    def VAE_loss(self, mse_loss, mu, logvar):
        MSE = mse_loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD

    def fit(self, xtrain: np.ndarray, lambda_reg=1e-6, epochs=5, lr=1e-3, batch_size=32): #batch_size=32
        train_set = VAEDataset(xtrain, with_target=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lambda_reg)
        criterion = nn.MSELoss()

        # Train the VAE with the new prior
        ELBO = np.zeros((epochs, 1))
        print("Start training of Variational Autoencoder...")
        for epoch in range(epochs):
            # Initialize the losses
            train_loss = 0
            train_loss_num = 0
            # Train for all the batches
            for data, _ in train_loader:
                data = data.view(data.shape[0], -1)
                # forward pass
                MU_X_eval, LOG_VAR_X_eval, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval = self(data)
                reconstruction = MU_X_eval
                mse_loss = criterion(reconstruction, data)
                loss = self.VAE_loss(mse_loss, MU_Z_eval, LOG_VAR_Z_eval)
                # Update the parameters
                optimizer.zero_grad()
                # Compute the loss
                loss.backward()
                # Update the parameters
                optimizer.step()
                # Collect the ways
                train_loss += loss.item()
                train_loss_num += 1

            ELBO[epoch] = train_loss / train_loss_num
            if epoch % 10 == 0:
                print("[Epoch: {}/{}] [objective: {:.3f}]".format(epoch, epochs, ELBO[epoch, 0]))

            ELBO_train = ELBO[epoch, 0].round(2)
            print("[ELBO train: " + str(ELBO_train) + "]")
        del MU_X_eval, MU_Z_eval, Z_ENC_eval
        del LOG_VAR_X_eval, LOG_VAR_Z_eval

        self.save()
        print("... finished training of Variational Autoencoder.")

    def load(self, input_shape):
        cache_path = get_home()
        load_path = os.path.join(cache_path, "{}_{}.{}".format(self._data_name, input_shape, "pt"))
        self.load_state_dict(torch.load(load_path))
        self.eval()
        return self

    def save(self):
        cache_path = get_home()
        save_path = os.path.join(cache_path, "{}_{}.{}".format(self._data_name, self._input_dim, "pt"))
        torch.save(self.state_dict(), save_path)

class CCHVAE(RecourseMethod):

    _DEFAULT_HYPERPARAMS = {"data_name": None, "n_search_samples": 500, "p_norm": 1, "step": 0.1, "max_iter": 2000,
                            "clamp": True, "binary_cat_features": True, "vae_params": {"layers": [10, 5, 10], "train": True,
                                                                                       "lambda_reg": 1e-6, "epochs": 20,
                                                                                       "lr": 1e-3, "batch_size": 32}} #"batch_size": 32

    def __init__(self, counterfactual):
        data = counterfactual.data
        model = counterfactual.model
        self.undesired_label = data.undesired_class
        factuals = counterfactual.ioi.normal_x_df
        cchvae_data = MyOwnDataSet(data)
        cchvae_model = MyOwnModel(cchvae_data, model)
        super().__init__(cchvae_model)
        self._params = self._DEFAULT_HYPERPARAMS
        self._params['vae_params']['layers'] = [len(cchvae_data.processed_features), 15, 10, 15, len(cchvae_data.processed_features)]
        self._n_search_samples = self._params["n_search_samples"]
        self._p_norm = self._params["p_norm"]
        self._step = self._params["step"]
        self._max_iter = self._params["max_iter"]
        self._clamp = self._params["clamp"]
        vae_params = self._params["vae_params"]
        self._generative_model = self._load_vae(self._mlmodel.data.df, vae_params, self._mlmodel, self._params["data_name"])
        start_time = time.time()
        cfs = self.get_counterfactuals(factuals)
        end_time = time.time()
        run_time = end_time - start_time
        cfs = cfs[data.processed_features]
        cfs = cfs.values[0]
        self.normal_x_cf, self.run_time = cfs, run_time

    def _load_vae(self, data: pd.DataFrame, vae_params, mlmodel: MLModel, data_name: str) -> VariationalAutoencoder:
        generative_model = VariationalAutoencoder(data_name, vae_params["layers"])
        if vae_params["train"]:
            generative_model = train_variational_autoencoder(generative_model, mlmodel.data, mlmodel.feature_input_order, lambda_reg=vae_params["lambda_reg"], epochs=vae_params["epochs"],
                                                             lr=vae_params["lr"], batch_size=vae_params["batch_size"])
        else:
            try:
                generative_model.load(data.shape[1] - 1)
            except FileNotFoundError as exc:
                raise FileNotFoundError("Loading of Autoencoder failed. {}".format(str(exc)))
        return generative_model

    def _hyper_sphere_coordindates(self, instance, high: int, low: int):
        """
        :param n_search_samples: int > 0
        :param instance: numpy input point array
        :param high: float>= 0, h>l; upper bound
        :param low: float>= 0, l<h; lower bound
        :param p: float>= 1; norm
        :return: candidate counterfactuals & distances
        """
        delta_instance = np.random.randn(self._n_search_samples, instance.shape[1])
        dist = (np.random.rand(self._n_search_samples) * (high - low) + low)
        norm_p = LA.norm(delta_instance, ord=self._p_norm, axis=1)
        d_norm = np.divide(dist, norm_p).reshape(-1, 1)  # rescale/normalize factor
        delta_instance = np.multiply(delta_instance, d_norm)
        candidate_counterfactuals = instance + delta_instance
        return candidate_counterfactuals, dist

    def _counterfactual_search(self, step: int, factual: torch.Tensor, cat_features_indices) -> pd.DataFrame:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # init step size for growing the sphere
        low = 0
        high = step
        # counter
        count = 0
        counter_step = 1
        torch_fact = torch.from_numpy(factual).to(device)
        # get predicted label of instance
        instance_label = np.argmax(self._mlmodel.predict_proba(torch_fact.float()), axis=1)

        # vectorize z
        z = self._generative_model.encode(torch_fact.float())[0].cpu().detach().numpy()
        z_rep = np.repeat(z.reshape(1, -1), self._n_search_samples, axis=0)

        candidate_dist = []
        x_ce: Union[np.ndarray, torch.Tensor] = np.array([])
        while count <= self._max_iter or len(candidate_dist) <= 0:
            count = count + counter_step
            if count > self._max_iter:
                print("No counterfactual example found")
                return x_ce[0]

            # STEP 1 -- SAMPLE POINTS on hyper sphere around instance
            latent_neighbourhood, _ = self._hyper_sphere_coordindates(z_rep, high, low)
            torch_latent_neighbourhood = (torch.from_numpy(latent_neighbourhood).to(device).float())
            x_ce = self._generative_model.decode(torch_latent_neighbourhood)[0]
            x_ce = reconstruct_encoding_constraints(x_ce, cat_features_indices, self._params["binary_cat_features"])
            x_ce = x_ce.detach().cpu().numpy()
            x_ce = x_ce.clip(0, 1) if self._clamp else x_ce

            # STEP 2 -- COMPUTE l1 & l2 norms
            if self._p_norm == 1:
                distances = np.abs((x_ce - torch_fact.cpu().detach().numpy())).sum(axis=1)
            elif self._p_norm == 2:
                distances = LA.norm(x_ce - torch_fact.cpu().detach().numpy(), axis=1)
            else:
                raise ValueError("Possible values for p_norm are 1 or 2")

            # counterfactual labels
            y_candidate = np.argmax(self._mlmodel.predict_proba(torch.from_numpy(x_ce).float()), axis=1)
            indeces = np.where(y_candidate != instance_label)
            candidate_counterfactuals = x_ce[indeces]
            candidate_dist = distances[indeces]
            # no candidate found & push search range outside
            if len(candidate_dist) == 0:
                low = high
                high = low + step
            elif len(candidate_dist) > 0:
                # certain candidates generated
                min_index = np.argmin(candidate_dist)
                print("Counterfactual example found")
                return candidate_counterfactuals[min_index]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        # factuals = self._mlmodel.get_ordered_features(factuals)
        encoded_feature_names = self._mlmodel.data.encoder.get_feature_names(self._mlmodel.data.categorical)
        cat_features_indices = [factuals.columns.get_loc(feature) for feature in encoded_feature_names]
        df_cfs = factuals.apply(lambda x: self._counterfactual_search(self._step, x.reshape((1, -1)), cat_features_indices), raw=True, axis=1)
        cf_pred = self._mlmodel.predict(df_cfs)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        if cf_pred == self.undesired_label:
            print(f'CCHVAE error: Counterfactual has same label as the IOI')
        return df_cfs

def order_data(feature_order, df):
    """
    Restores the correct input feature order for the ML model
    """
    return df[feature_order]

def get_home(models_home=None):
    """
    Return a path to the cache directory for trained autoencoders.
    """
    if models_home is None:
        models_home = os.environ.get("CF_MODELS", os.path.join("~", "carla", "models", "autoencoders"))
    models_home = os.path.expanduser(models_home)
    if not os.path.exists(models_home):
        os.makedirs(models_home)
    return models_home

def train_variational_autoencoder(vae, data, input_order, lambda_reg=1e-6, epochs=5, lr=1e-3, batch_size=32): #batch_size=32
    df_dataset = data.df[input_order + [data.target]]
    vae.fit(df_dataset.values, lambda_reg, epochs, lr, batch_size)
    vae.eval()
    return vae

def reconstruct_encoding_constraints(x: torch.Tensor, feature_pos, binary_cat: bool) -> torch.Tensor:
    """
    Reconstructing one-hot-encoded data, such that its values are either 0 or 1,
    and features do not contradict (e.g., sex_female = 1, sex_male = 1)
    """
    x_enc = x.clone()
    if binary_cat:
        for pos in feature_pos:
            x_enc[:, pos] = torch.round(x_enc[:, pos])
    else:
        binary_pairs = list(zip(feature_pos[:-1], feature_pos[1:]))[0::2]
        for pair in binary_pairs:
            # avoid overwritten inconsistent results
            temp = (x_enc[:, pair[0]] >= x_enc[:, pair[1]]).float()
            x_enc[:, pair[1]] = (x_enc[:, pair[0]] < x_enc[:, pair[1]]).float()
            x_enc[:, pair[0]] = temp
            if (x_enc[:, pair[0]] == x_enc[:, pair[1]]).any():
                raise ValueError("Reconstructing encoded features lead to an error. Feature {} and {} have the same value".format(pair[0], pair[1]))
    return x_enc
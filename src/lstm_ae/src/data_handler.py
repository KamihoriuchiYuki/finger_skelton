import data_processing as dp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import pickle

class DataHandler:
    def __init__(self, data_path, scaler_rs4=None, scaler_sg1=None):
        self.data_path = os.path.expanduser(data_path)
        self.df = self._load_data()
        self.scaler_rs4 = scaler_rs4
        self.scaler_sg1 = scaler_sg1
        self._filter_data()
        self._scale_data()
        self.xtrain = None
        self.xtest  = None
        self.ytrain = None
        self.ytest  = None

    def _load_data(self):
        return dp.get_data(self.data_path)

    def _filter_data(self):
        self.df['rs4_filtered'] = self.df['rs4'].apply(lambda x: x if -90 <= x <= 150 else np.nan)
        self.df['sg1_filtered'] = self.df['sg1'].apply(lambda x: x if 100 <= x <= 600 else np.nan)

    def _scale_data(self):
        if self.scaler_rs4 is None:
            self.scaler_rs4 = StandardScaler()
            self.scaler_rs4.fit(self.df['rs4_filtered'].values.reshape(-1, 1))
        if self.scaler_sg1 is None:
            self.scaler_sg1 = StandardScaler()
            self.scaler_sg1.fit(self.df['sg1_filtered'].values.reshape(-1, 1))
        
        self.df['rs4_scaled'] = self.scaler_rs4.transform(self.df['rs4_filtered'].values.reshape(-1, 1))
        self.df['sg1_scaled'] = self.scaler_sg1.transform(self.df['sg1_filtered'].values.reshape(-1, 1))


    def get_scalers(self):
        return self.scaler_rs4, self.scaler_sg1
    
    def split_data(self, test_size=0.1):
        self.xtrain, self.xtest = dp.split_data(self.df['t'], self.df['rs4_scaled'], test_size=test_size)
        self.ytrain, self.ytest = dp.split_data(self.df['t'], self.df['sg1_scaled'], test_size=test_size)

    def introduce_missingness(self, x_missing_rate=0.0, y_missing_rate=0.0):
        self.xtrain['data_missing'] = dp.introduce_missingness(self.xtrain['data'], missing_rate=x_missing_rate)
        self.xtrain['if_missing'] = self.xtrain['data_missing'].isnull()
        self.xtest['data_missing'] = dp.introduce_missingness(self.xtest['data'], missing_rate=x_missing_rate)

        self.ytrain['data_missing'] = dp.introduce_missingness(self.ytrain['data'], missing_rate=y_missing_rate)
        self.ytrain['if_missing'] = self.ytrain['data_missing'].isnull()
        self.ytest['data_missing'] = dp.introduce_missingness(self.ytest['data'], missing_rate=y_missing_rate)

    def make_sequence(self, timesteps):
        X_train = dp.make_sequence(self.xtrain['data_missing'], timesteps)
        X_test  = dp.make_sequence(self.xtest['data_missing'], timesteps)
        Y_train = dp.make_sequence(self.ytrain['data_missing'], timesteps)
        Y_test  = dp.make_sequence(self.ytest['data_missing'], timesteps)
        
        X_train = np.expand_dims(X_train, axis=-1)
        X_test  = np.expand_dims(X_test, axis=-1)
        Y_train = np.expand_dims(Y_train, axis=-1)
        Y_test  = np.expand_dims(Y_test, axis=-1)
        
        return X_train, X_test, Y_train, Y_test

def main():
    # Define file paths
    path1 = "~/Sensor-Glove/src/data_handler/data/0813_1559_ind_sample.csv"
    path2 = "~/Sensor-Glove/src/data_handler/data/other_sample.csv"

    # Initialize the first data handler
    data1 = DataHandler(path1)
    data1.split_data(test_size=0.1)
    data1.introduce_missingness(x_missing_rate=0.5, y_missing_rate=0.5)

    # Initialize the second data handler using the same scalers as the first
    data2 = DataHandler(path2, scaler_rs4=data1.scaler_rs4, scaler_sg1=data1.scaler_sg1)
    data2.split_data(test_size=0.1)
    data2.introduce_missingness(x_missing_rate=0.5, y_missing_rate=0.5)

    # Concatenate the training and testing data from both data handlers
    X_train = np.concatenate([data1.make_sequence(timesteps=10)[0], data2.make_sequence(timesteps=10)[0]])
    Y_train = np.concatenate([data1.make_sequence(timesteps=10)[2], data2.make_sequence(timesteps=10)[2]])

    X_test  = np.concatenate([data1.make_sequence(timesteps=10)[1], data2.make_sequence(timesteps=10)[1]])
    Y_test  = np.concatenate([data1.make_sequence(timesteps=10)[3], data2.make_sequence(timesteps=10)[3]])

    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

    # Proceed with LSTM model training using X_train, Y_train, X_test, and Y_test

if __name__ == "__main__":
    main()

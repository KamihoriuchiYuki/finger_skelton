import data_processor as dp
import lstm_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import os
import time
import data_handler as dh
from tensorflow.keras.utils import plot_model


# Set parameters
params = {
    "input_dim": 1,
    "latent_dim": 20,
    "timesteps": 20,
    "train_size": None,
    "epochs": 200,
    "batch_size": 64,
    "x_missing_rate": 0.3,
    "y_missing_rate": 0.0,
    "train_data_paths": [
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_220_350_1.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_220_350_2.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_300_400.csv",
        "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_1.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_2.csv",
        "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_4.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_5.csv",
        "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_6.csv",
        "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_7.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_8.csv",
        "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_9.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_10.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_11.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_2.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_6.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_7.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_9.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_10.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_11.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_12.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_14.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_450_550.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_500_620_2.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_500_620_4.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_2.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_5.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_6.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_8.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_10.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_11.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_12.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_1.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_2.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_3.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_4.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_5.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_6.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_7.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_8.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_9.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_10.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_11.csv",
        # "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_12.csv"
    ]
}

input_dim    = params["input_dim"]
latent_dim   = params["latent_dim"]
timesteps    = params["timesteps"]
epochs       = params["epochs"]
batch_size   = params["batch_size"]
x_missing_rate = params["x_missing_rate"]
y_missing_rate = params["y_missing_rate"]
train_data_paths     = params["train_data_paths"]

## Initialize variables to hold concatenated data
X_train_list = []
Y_train_list = []
X_test_list  = []
Y_test_list  = []
X_train_missing_list = []
Y_train_missing_list = []
X_test_missing_list  = []
Y_test_missing_list  = []

# Initialize the first DataHandler instance to fit the scalers
dh1 = dh.DataHandler(train_data_paths[0])
std_scaler_rs4, std_scaler_sg1 = dh1.get_scalers()

# Process each dataset in the train_data_paths list
for data_path in train_data_paths:
    data_handler = dh.DataHandler(data_path, scaler_rs4=std_scaler_rs4, scaler_sg1=std_scaler_sg1)
    data_handler.split_data(test_size=0.1)
    data_handler.introduce_missingness(x_missing_rate=x_missing_rate, y_missing_rate=y_missing_rate)
    X_train, X_test, Y_train, Y_test, X_train_missing, X_test_missing, Y_train_missing, Y_test_missing = data_handler.make_sequence(timesteps, is_random=False)

    # Collect the processed data
    X_train_list.append(X_train)
    Y_train_list.append(Y_train)
    X_test_list.append(X_test)
    Y_test_list.append(Y_test)
    X_train_missing_list.append(X_train_missing)
    Y_train_missing_list.append(Y_train_missing)
    X_test_missing_list.append(X_test_missing)
    Y_test_missing_list.append(Y_test_missing)

# Concatenate all the data from the list
X_train_raw = np.concatenate(X_train_list, axis=0)
Y_train_raw = np.concatenate(Y_train_list, axis=0)
X_test_raw  = np.concatenate(X_test_list,  axis=0)
Y_test_raw  = np.concatenate(Y_test_list,  axis=0)
X_train_missing = np.concatenate(X_train_missing_list, axis=0)
Y_train_missing = np.concatenate(Y_train_missing_list, axis=0)
X_test_missing  = np.concatenate(X_test_missing_list,  axis=0)
Y_test_missing  = np.concatenate(Y_test_missing_list,  axis=0)

# Apply np.nan_to_num to handle NaN values
X_train_raw_zero = np.nan_to_num(X_train_raw, nan=0)
X_test_raw_zero  = np.nan_to_num(X_test_raw,  nan=0)
Y_train_raw_zero = np.nan_to_num(Y_train_raw, nan=0)
Y_test_raw_zero  = np.nan_to_num(Y_test_raw,  nan=0)
X_train_missing_zero = np.nan_to_num(X_train_missing, nan=0)
X_test_missing_zero  = np.nan_to_num(X_test_missing,  nan=0)
Y_train_missing_zero = np.nan_to_num(Y_train_missing, nan=0)
Y_test_missing_zero  = np.nan_to_num(Y_test_missing,  nan=0)

## Choose the training data
X_train_zero = X_train_raw_zero
X_test_zero  = X_test_raw_zero
Y_train_zero = Y_train_missing_zero
Y_test_zero  = Y_test_missing_zero

# make model
model=lstm_model.lstm_ae_2i2o(input_dim, latent_dim, timesteps)
# model=lstm_model.lstm_ae_simple(input_dim, latent_dim, timesteps)
# model=lstm_model.autoencoder(input_dim, latent_dim, timesteps)
model.summary()
model.compile(optimizer='adam', loss=lstm_model.masked_mse_loss)
# history = model.fit([X_train_missing_zero, Y_train_missing_zero], [X_train_missing_zero, Y_train_missing_zero], epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=([X_train_missing_zero, Y_train_missing_zero], [X_train_zero, Y_train_zero]))
history = model.fit([X_train_missing_zero, Y_train_missing_zero], [X_train_raw_zero, Y_train_missing_zero], epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=([X_train_missing_zero, Y_train_missing_zero], [X_train_raw_zero, Y_train_missing_zero]))

# predict with model
X_hat_train, Y_hat_train = model.predict([X_train_missing_zero,Y_train_missing_zero])
X_hat_test,  Y_hat_test  = model.predict([X_test_missing_zero, Y_test_missing_zero])
# print("XY_hat_train.shape : ", XY_hat_train.shape)

# save
save_dir = dp.prep_save_dir()
params['train_size'] = X_train_zero.shape[0]
dp.save_params(params, save_dir, 'params')
dp.save_model(model, save_dir, 'lstm_ae_2i2o')
# dp.save_model(model, save_dir, 'autoencoder')
# plot_model(model, to_file=os.path.join(save_dir, 'model.png'), show_shapes=True)
pickle.dump(std_scaler_rs4, open(os.path.join(save_dir, 'std_scaler_rs4.pkl'), 'wb'))
pickle.dump(std_scaler_sg1, open(os.path.join(save_dir, 'std_scaler_sg1.pkl'), 'wb'))

# inverse transform
X_train_raw = std_scaler_rs4.inverse_transform(X_train_raw[:,-1,0].reshape(-1, 1)).flatten()
X_test_raw  = std_scaler_rs4.inverse_transform(X_test_raw [:,-1,0].reshape(-1, 1)).flatten()
X_hat_train  = std_scaler_rs4.inverse_transform(X_hat_train[:,-1,0].reshape(-1, 1)).flatten()
X_hat_test   = std_scaler_rs4.inverse_transform(X_hat_test [:,-1,0].reshape(-1, 1)).flatten()
X_train_missing = std_scaler_rs4.inverse_transform(X_train_missing[:,-1,0].reshape(-1, 1)).flatten()
X_test_missing  = std_scaler_rs4.inverse_transform(X_test_missing [:,-1,0].reshape(-1, 1)).flatten()

Y_train_raw = std_scaler_sg1.inverse_transform(Y_train_raw[:,-1,0].reshape(-1, 1)).flatten()
Y_test_raw  = std_scaler_sg1.inverse_transform(Y_test_raw [:,-1,0].reshape(-1, 1)).flatten()
Y_hat_train = std_scaler_sg1.inverse_transform(Y_hat_train[:,-1,0].reshape(-1, 1)).flatten()
Y_hat_test  = std_scaler_sg1.inverse_transform(Y_hat_test [:,-1,0].reshape(-1, 1)).flatten()
Y_train_missing = std_scaler_sg1.inverse_transform(Y_train_missing[:,-1,0].reshape(-1, 1)).flatten()
Y_test_missing  = std_scaler_sg1.inverse_transform(Y_test_missing [:,-1,0].reshape(-1, 1)).flatten()

# Plot training and validation loss
history_data  = [np.array(history.history['loss']), np.array(history.history['val_loss'])]
history_label = ['Train loss', 'Valid loss']
dp.save_plot(history_data, history_label, save_dir, 'loss')

# Plot original training data (RS and SG with missing values)
train_data  = [X_train_missing, Y_train_raw]
train_label = ['RS Train', 'SG Train']
dp.save_plot(train_data, train_label, save_dir, 'train_data')

# Evaluate RS values for training data (Actual vs Predicted)
evaluate_data_train_rs  = [X_train_missing, X_hat_train]
evaluate_label_train_rs = ['RS Actual Train missing', 'RS Predicted Train']
dp.save_plot(evaluate_data_train_rs, evaluate_label_train_rs, save_dir, 'rs_prediction_train_missing')

# Evaluate RS values for testing data (Actual vs Predicted)
evaluate_data_test_rs  = [X_test_raw, X_hat_test]
evaluate_label_test_rs = ['RS Actual Test raw', 'RS Predicted Test']
dp.save_plot(evaluate_data_test_rs, evaluate_label_test_rs, save_dir, 'rs_prediction_test_raw')

# Evaluate RS values for testing data (Actual vs Predicted)
evaluate_data_test_rs  = [X_test_missing, X_hat_test]
evaluate_label_test_rs = ['RS Actual Test missing', 'RS Predicted Test']
dp.save_plot(evaluate_data_test_rs, evaluate_label_test_rs, save_dir, 'rs_prediction_test_missing')

# Evaluate SG values for training data (Actual vs Predicted)
evaluate_data_train_sg  = [Y_train_raw, Y_hat_train]
evaluate_label_train_sg = ['SG Actual Train', 'SG Predicted Train']
dp.save_plot(evaluate_data_train_sg, evaluate_label_train_sg, save_dir, 'sg_prediction_train')

# Evaluate SG values for training data (Actual vs Predicted)
evaluate_data_train_sg  = [Y_train_raw, Y_hat_train]
evaluate_label_train_sg = ['SG Actual Train', 'SG Predicted Train']
dp.save_plot(evaluate_data_train_sg, evaluate_label_train_sg, save_dir, 'sg_prediction_train')

# Evaluate SG values for testing data (Actual vs Predicted)
evaluate_data_test_sg  = [Y_test_raw, Y_hat_test]
evaluate_label_test_sg = ['SG Actual Test', 'SG Predicted Test']
dp.save_plot(evaluate_data_test_sg, evaluate_label_test_sg, save_dir, 'sg_prediction_test')

# Evaluate SG values for testing data (Actual vs Predicted)
evaluate_data_test_sg  = [Y_test_missing, Y_hat_test]
evaluate_label_test_sg = ['SG Actual Test missing', 'SG Predicted Test']
dp.save_plot(evaluate_data_test_sg, evaluate_label_test_sg, save_dir, 'sg_prediction_test_missing')

# Beep sound when finished
for i in range(1):
    print("\a")
    time.sleep(1)

import data_processor as dp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lstm_model
from sklearn.preprocessing import StandardScaler
import os
import math
import time
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle
import data_handler as dh

# Get path of pre-trained model
first_train_path = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/results/0823/1716")
epochs = 150
x_missing_rate = 0.4
y_missing_rate = 1.0

# Set folder name for saving results
save_dir = dp.prep_save_dir()
checkpoint_dir = os.path.join(save_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Get parameters from param file
param_path = os.path.join(first_train_path, "params.json")
params = dp.read_json(param_path)
train_data_paths = params["train_data_paths"]  # Multiple data paths
timesteps = params["timesteps"]
input_dim = params["input_dim"]
latent_dim = params["latent_dim"]
batch_size = params["batch_size"]
params['epochs'] = epochs
params['x_missing_rate'] = x_missing_rate
params['y_missing_rate'] = y_missing_rate

# Load the pre-trained model
model_path = os.path.join(first_train_path, "lstm_ae_2i2o.keras")
# model_path = os.path.join(first_train_path, "autoencoder.keras")
model = load_model(model_path, compile=False)
model.load_weights(model_path)
model.compile(optimizer='adam', loss=lstm_model.masked_mse_loss)
params['model_path'] = model_path

# Load scalers from the first training
std_scaler_rs4 = pickle.load(open(os.path.join(first_train_path, "std_scaler_rs4.pkl"), 'rb'))
std_scaler_sg1 = pickle.load(open(os.path.join(first_train_path, "std_scaler_sg1.pkl"), 'rb'))

# Initialize lists to hold concatenated data
X_train_list = []
Y_train_list = []
X_test_list  = []
Y_test_list  = []
X_train_missing_list = []
Y_train_missing_list = []
X_test_missing_list  = []
Y_test_missing_list  = []

# Process each dataset in the train_data_paths list
for data_path in train_data_paths:
    # Initialize DataHandler with pre-loaded scalers
    data_handler = dh.DataHandler(data_path, scaler_rs4=std_scaler_rs4, scaler_sg1=std_scaler_sg1)
    
    # Split data and introduce missingness
    data_handler.split_data(test_size=0.1)
    data_handler.introduce_missingness(x_missing_rate=x_missing_rate, y_missing_rate=y_missing_rate)
    
    # Generate sequences for LSTM
    X_train, X_test, Y_train, Y_test, X_train_missing, X_test_missing, Y_train_missing, Y_test_missing = data_handler.make_sequence(timesteps, is_random=False)
    
    # Append to the lists
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
X_test_raw  = np.concatenate(X_test_list, axis=0)
Y_test_raw  = np.concatenate(Y_test_list, axis=0)
X_train_missing = np.concatenate(X_train_missing_list, axis=0)
Y_train_missing = np.concatenate(Y_train_missing_list, axis=0)
X_test_missing  = np.concatenate(X_test_missing_list, axis=0)
Y_test_missing  = np.concatenate(Y_test_missing_list, axis=0)

# Apply np.nan_to_num to handle NaN values
X_train_raw_zero = np.nan_to_num(X_train_raw, nan=0)
X_test_raw_zero  = np.nan_to_num(X_test_raw, nan=0)
Y_train_raw_zero = np.nan_to_num(Y_train_raw, nan=0)
Y_test_raw_zero  = np.nan_to_num(Y_test_raw, nan=0)

X_train_missing_zero = np.nan_to_num(X_train_missing, nan=0)
X_test_missing_zero  = np.nan_to_num(X_test_missing, nan=0)
Y_train_missing_zero = np.nan_to_num(Y_train_missing, nan=0)
Y_test_missing_zero  = np.nan_to_num(Y_test_missing, nan=0)

## Choose the training data
X_train_zero = X_train_missing_zero
X_test_zero  = X_test_missing_zero
Y_train_zero = Y_train_raw_zero
Y_test_zero  = Y_test_raw_zero

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = os.path.join(checkpoint_dir, "weights_epoch_{epoch:04d}.weights.h5")
n_batches = len(X_train_zero) / batch_size
n_batches = math.ceil(n_batches)  # Round up the number of batches to the nearest whole integer

# Create a callback that saves the model's weights every 10 batches
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=10 * n_batches
)

# Save the initial weights
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model
history = model.fit([X_train_missing_zero, Y_train_missing_zero], 
                    [X_train_raw_zero, Y_train_missing_zero], 
                    epochs=epochs, 
                    batch_size=batch_size,
                    shuffle=True, 
                    validation_data=([X_train_missing_zero, Y_train_missing_zero], [X_train_raw_zero, Y_train_missing_zero]),
                    callbacks=[cp_callback])

# Predict with the model
X_hat_train, Y_hat_train = model.predict([X_train_missing_zero, Y_train_missing_zero])
X_hat_test,  Y_hat_test  = model.predict([X_test_missing_zero, Y_test_missing_zero])

# Save the results
params['train_size'] = X_train_zero.shape[0]
dp.save_params(params, save_dir, 'params')
dp.save_model(model, save_dir, 'lstm_ae_2i2o')
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

# comment
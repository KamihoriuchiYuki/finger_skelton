import data_processor as dp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lstm_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import math
import time
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle
import data_handler as dh

# get path of pre trained model
first_train_path = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/results/0816/1602")
epochs = 100

# set folder name
save_dir = dp.prep_save_dir()
checkpoint_dir = os.path.join(save_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
# get params
param_path = os.path.join(first_train_path, "params")
params = dp.read_json(param_path)
data_path = params["data_path"]
timesteps = params["timesteps"]
input_dim = params["input_dim"]
latent_dim = params["latent_dim"]
# epochs = params["epochs"]
batch_size = params["batch_size"]
params['epochs'] = epochs

# get model
model_path = os.path.join(first_train_path, "lstm_ae_2i2o.keras")
model = load_model(model_path, compile=False)
model.load_weights(model_path)
model.compile(optimizer='adam', loss=lstm_model.masked_mse_loss)
params['model_path'] = model_path

# # freeze layers
# freeze_layer = [0,2,4,8,10,12,14,16]
# for i in freeze_layer:
#     model.layers[i].trainable = False

# get data from csv
file_path = os.path.expanduser(data_path)

df = dp.get_data(file_path)
df['rs4_filtered'] = df['rs4'].apply(lambda x: x if -90 <= x <= 150 else np.nan)
# df['rs4_lowpass'] = dp.fft_lowpass(df['rs4_filtered'], 5)
df['sg1_filtered'] = df['sg1'].apply(lambda x: x if 100 <= x <= 600 else np.nan)

# scale data
# std_scaler = StandardScaler()
# std_scaler.fit(df)
# df_std = pd.DataFrame(std_scaler.transform(df), columns=df.columns)
std_scaler_rs4 = pickle.load(open(os.path.join(first_train_path, "std_scaler_rs4.pkl"), 'rb'))
std_scaler_sg1 = pickle.load(open(os.path.join(first_train_path, "std_scaler_sg1.pkl"), 'rb'))
scaled_rs4 = std_scaler_rs4.transform(df['rs4_filtered'].values.reshape(-1, 1))
scaled_sg1 = std_scaler_sg1.transform(df['sg1_filtered'].values.reshape(-1, 1))
df['rs4_scaled'] = scaled_rs4
df['sg1_scaled'] = scaled_sg1



# split data
df_xtrain, df_xtest = dp.split_data(df['t'], df['rs4_scaled'], test_size=0.1)
df_ytrain, df_ytest = dp.split_data(df['t'], df['sg1_scaled'], test_size=0.1)
# df_xtrain, df_xtest = dp.split_data(df_std['t'], df_std['rs4_lowpass'], test_size=0.1)
# df_xtrain, df_xtest = dp.split_data(df_std['t'], df_std['rs4_filtered'], test_size=0.1)
# df_ytrain, df_ytest = dp.split_data(df_std['t'], df_std['sg1_filtered'], test_size=0.1)
# df_xtrain, df_xtest = dp.split_data(df_std['t'], df_std['rs4'], test_size=0.1)
# df_ytrain, df_ytest = dp.split_data(df_std['t'], df_std['sg1'], test_size=0.1)

# introduce missingness
missing_rate = 0.0
df_xtrain['data_missing'] = dp.introduce_missingness(df_xtrain['data'], missing_rate = 0.5)
df_xtrain['if_missing']   = df_xtrain['data_missing'].isnull() # label missing data
df_xtest ['data_missing'] = dp.introduce_missingness(df_xtest['data'],  missing_rate = 0.5)

df_ytrain['data_missing'] = dp.introduce_missingness(df_ytrain['data'], missing_rate)
df_ytrain['if_missing']   = df_ytrain['data_missing'].isnull() # label missing data
df_ytest ['data_missing'] = dp.introduce_missingness(df_ytest['data'],  missing_rate)

# make sequence for LSTM
X_train = dp.make_sequence(df_xtrain['data_missing'], timesteps)
X_test  = dp.make_sequence(df_xtest['data_missing'],  timesteps)
Y_train = dp.make_sequence(df_ytrain['data'], timesteps)
Y_test  = dp.make_sequence(df_ytest['data'],  timesteps)
# 次元を追加
X_train = np.expand_dims(X_train, axis=-1)
X_test  = np.expand_dims(X_test,  axis=-1)
Y_train = np.expand_dims(Y_train, axis=-1)
Y_test  = np.expand_dims(Y_test,  axis=-1)
print(f"X_train.shape : {X_train.shape}, Y_train.shape : {Y_train.shape}")  

# concatenate X and Y (input dimention of LSTM will be 2)
# XY_train = np.concatenate([X_train, Y_train], axis=2)
# XY_test  = np.concatenate([X_test , Y_test ], axis=2)
# delete data if the last data is missing (because we want to predict the last data)
X_train_zero = np.nan_to_num(X_train, nan=0)
X_test_zero  = np.nan_to_num(X_test , nan=0)
Y_train_zero = np.nan_to_num(Y_train, nan=0)
Y_test_zero  = np.nan_to_num(Y_test , nan=0)


# Include the epoch in the file name (uses `str.format`)
checkpoint_path = os.path.join(checkpoint_dir, "weights_epoch_{epoch:04d}.weights.h5")
# checkpoint_path = dp._save_path("weights_epoch_{epoch:04d}.weights.h5")
# checkpoint_path = os.path.join(save_dir, "weights_epoch_{epoch:04d}.weights.h5")
checkpoint_dir = os.path.dirname(checkpoint_path)
n_batches = len(X_train_zero) / batch_size
n_batches = math.ceil(n_batches)    # round up the number of batches to the nearest whole integer

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=10*n_batches)

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# make model
history = model.fit([X_train_zero, Y_train_zero], 
                    [X_train_zero, Y_train_zero], 
                    epochs=epochs, 
                    batch_size=batch_size,
                    shuffle=True, 
                    validation_data=([X_train_zero, Y_train_zero], [X_train_zero, Y_train_zero]),
                    callbacks=[cp_callback])

# predict with model
X_hat_train, Y_hat_train = model.predict([X_train_zero,Y_train_zero])
X_hat_test,  Y_hat_test  = model.predict([X_test_zero, Y_test_zero])
# print("XY_hat_train.shape : ", XY_hat_train.shape)

# save
params['train_size'] = X_train_zero.shape[0]
dp.save_params(params, save_dir, 'params')
dp.save_model(model, save_dir,'lstm_ae_2i2o')
pickle.dump(std_scaler_rs4, open(os.path.join(save_dir, 'std_scaler_rs4.pkl'), 'wb'))
pickle.dump(std_scaler_sg1, open(os.path.join(save_dir, 'std_scaler_sg1.pkl'), 'wb'))

history_data  = [np.array(history.history['loss']), np.array(history.history['val_loss'])]
history_label = ['Train loss', 'Valid loss']
dp.save_plot(history_data, history_label, save_dir, 'loss')

train_data  = [df_xtrain['data_missing'].to_numpy(), df_ytrain['data'].to_numpy()]
train_label = ['rs train', 'sg train']   
dp.save_plot(train_data, train_label, save_dir, 'train_data')

evaluate_data_train  = [df_xtrain['data_missing'][timesteps:].to_numpy(), X_hat_train[1:,-1,0]]
evaluate_label_train = ['rs actual', 'rs predict']
dp.save_plot(evaluate_data_train, evaluate_label_train, save_dir, 'rs_prediction_train')

evaluate_data_test  = [df_xtest['data_missing'][timesteps:].to_numpy(), X_hat_test[1:,-1,0]]
evaluate_label_test = ['rs actual', 'rs predict']
dp.save_plot(evaluate_data_test, evaluate_label_test, save_dir, 'rs_prediction_test')

#evaluate y
evaluate_data_train  = [df_ytrain['data'][timesteps:].to_numpy(), Y_hat_train[1:,-1,0]]
evaluate_label_train = ['sg actual', 'sg predict']
dp.save_plot(evaluate_data_train, evaluate_label_train, save_dir, 'sg_prediction_train')

evaluate_data_test  = [df_ytest['data'][timesteps:].to_numpy(), Y_hat_test[1:,-1,0]]
evaluate_label_test = ['sg actual', 'sg predict']
dp.save_plot(evaluate_data_test, evaluate_label_test, save_dir, 'sg_prediction_test')

# beep sound when finished
for i in range(1):
    print("\a")
    time.sleep(1)
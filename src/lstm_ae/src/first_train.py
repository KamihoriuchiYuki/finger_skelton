import data_processor as dp
import lstm_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import os
import time


# set parameters
params = {
    "input_dim": 1,
    "latent_dim": 10,
    "timesteps": 10,
    "train_size": None,
    "epochs": 1,
    "batch_size": 64,
    "missing_rate": 0.0,
    "data_path": "~/Sensor-Glove/src/data_handler/data/0813_1559_ind_sample.csv"#ind_20min_0812_1436.csv
}

input_dim    = params["input_dim"]
latent_dim   = params["latent_dim"]
timesteps    = params["timesteps"]
epochs       = params["epochs"]
batch_size   = params["batch_size"]
missing_rate = params["missing_rate"]
data_path    = params["data_path"]

# get data from csv
file_path = os.path.expanduser(data_path)

# is_dummy = True
is_dummy = False
# 関数を呼び出してデータフレームを取得
if is_dummy:
    df_dummy = dp.make_dummy_data(random_factor=0.05, freq=(5, 5), amp=(1, -1), duration=15, step=0.001)
    df = df_dummy
    df['rs4'] = df_dummy['X']
    df['sg1'] = df_dummy['Y']
    df['sg1_filtered'] = df['sg1']
    df['rs4_filtered'] = df['rs4']
    params['data_path'] = 'dummy'
    
else:
    df = dp.get_data(file_path)
    df['rs4_filtered'] = df['rs4'].apply(lambda x: x if -90 <= x <= 150 else np.nan)
    # df['rs4_lowpass'] = dp.fft_lowpass(df['rs4_filtered'], 5)
    df['sg1_filtered'] = df['sg1'].apply(lambda x: x if 100 <= x <= 600 else np.nan)

# scale data
# std_scaler = StandardScaler()
# std_scaler.fit(df)
# df_std = pd.DataFrame(std_scaler.transform(df), columns=df.columns)
# Initialize scalers (same as used during training)
std_scaler_rs4 = StandardScaler()
std_scaler_sg1 = StandardScaler()
std_scaler_rs4.fit(df['rs4_filtered'].values.reshape(-1, 1))
std_scaler_sg1.fit(df['sg1_filtered'].values.reshape(-1, 1))
scaled_rs4 = std_scaler_rs4.transform(df['rs4_filtered'].values.reshape(-1, 1))
scaled_sg1 = std_scaler_sg1.transform(df['sg1_filtered'].values.reshape(-1, 1))
df['rs4_scaled'] = scaled_rs4
df['sg1_scaled'] = scaled_sg1

# split data
# df_xtrain, df_xtest = dp.split_data(df_std['t'], df_std['rs4_lowpass'], test_size=0.1)
# df_xtrain, df_xtest = dp.split_data(df_std['t'], df_std['rs4_filtered'], test_size=0.1)
# df_ytrain, df_ytest = dp.split_data(df_std['t'], df_std['sg1_filtered'], test_size=0.1)
df_xtrain, df_xtest = dp.split_data(df['t'], df['rs4_scaled'], test_size=0.1)
df_ytrain, df_ytest = dp.split_data(df['t'], df['sg1_scaled'], test_size=0.1)

# introduce missingness
df_xtrain['data_missing'] = dp.introduce_missingness(df_xtrain['data'], missing_rate=0.0)
df_xtrain['if_missing']   = df_xtrain['data_missing'].isnull() # label missing data
df_xtest ['data_missing'] = dp.introduce_missingness(df_xtest['data'],  missing_rate=0.0)

df_ytrain['data_missing'] = dp.introduce_missingness(df_ytrain['data'], missing_rate=1.0)
df_ytrain['if_missing']   = df_ytrain['data_missing'].isnull() # label missing data
df_ytest ['data_missing'] = dp.introduce_missingness(df_ytest['data'],  missing_rate=1.0)

# make sequence for LSTM
X_train = dp.make_sequence(df_xtrain['data_missing'], timesteps)
X_test  = dp.make_sequence(df_xtest['data_missing'],  timesteps)
Y_train = dp.make_sequence(df_ytrain['data_missing'], timesteps)
Y_test  = dp.make_sequence(df_ytest['data_missing'],  timesteps)
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

# mask_train = X_train_zero[:,-1,0] != 0
# mask_test  = X_test_zero [:,-1,0] != 0

# XY_true_zero = np.concatenate([X_train_zero, Y_train_zero], axis=2)

# X_train_zero_filtered = X_train_zero[mask_train]
# X_test_zero_filtered  = X_test_zero [mask_test]
# mask_train = Y_train_zero[:,-1,0] != 0
# mask_test  = Y_test_zero [:,-1,0] != 0
# Y_train_zero_filtered = Y_train_zero[mask_train]
# Y_test_zero_filtered  = Y_test_zero [mask_test]

# make model
# model=lstm_model.lstm_ae(input_dim, latent_dim, timesteps)
model=lstm_model.lstm_ae_2i2o(input_dim, latent_dim, timesteps)
model.summary()
model.compile(optimizer='adam', loss=lstm_model.masked_mse_loss_weight)
history = model.fit([X_train_zero, Y_train_zero], [X_train_zero, Y_train_zero], epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=([X_train_zero, Y_train_zero], [X_train_zero, Y_train_zero]))

# predict with model
X_hat_train, Y_hat_train = model.predict([X_train_zero,Y_train_zero])
X_hat_test,  Y_hat_test  = model.predict([X_test_zero, Y_test_zero])
# print("XY_hat_train.shape : ", XY_hat_train.shape)

# save
save_dir = dp.prep_save_dir()
params['train_size'] = X_train_zero.shape[0]
dp.save_params(params, save_dir, 'params')
dp.save_model(model, save_dir, 'lstm_ae_2i2o')
pickle.dump(std_scaler_rs4, open(os.path.join(save_dir, 'std_scaler_rs4.pkl'), 'wb'))
pickle.dump(std_scaler_sg1, open(os.path.join(save_dir, 'std_scaler_sg1.pkl'), 'wb'))

# inverse transform
X_train_zero = std_scaler_rs4.inverse_transform(X_train_zero[:,-1,0].reshape(-1, 1)).flatten()
X_hat_train  = std_scaler_rs4.inverse_transform(X_hat_train[:,-1,0].reshape(-1, 1)).flatten()
X_test_zero  = std_scaler_rs4.inverse_transform(X_test_zero[:,-1,0].reshape(-1, 1)).flatten()
X_hat_test   = std_scaler_rs4.inverse_transform(X_hat_test[:,-1,0].reshape(-1, 1)).flatten()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(X_train_zero, label='actual')
ax.plot(X_hat_train, label='predict')
ax2 = fig.add_subplot(1, 1, 1)
ax2.plot(X_test_zero, label='actual')
ax2.plot(X_hat_test, label='predict')
ax.legend()
plt.show()

# history_data  = [np.array(history.history['loss']), np.array(history.history['val_loss'])]
# history_label = ['Train loss', 'Valid loss']
# dp.save_plot(history_data, history_label, save_dir, 'loss')

# train_data  = [df_xtrain['data_missing'].to_numpy(), df_ytrain['data_missing'].to_numpy()]
# train_label = ['rs train', 'sg train']   
# dp.save_plot(train_data, train_label, save_dir, 'train_data')

# evaluate_data_train  = [df_xtrain['data_missing'][timesteps:].to_numpy(), X_hat_train[1:,-1,0]]
# evaluate_label_train = ['rs actual', 'rs predict']
# dp.save_plot(evaluate_data_train, evaluate_label_train, save_dir, 'rs_prediction_train')

# evaluate_data_test  = [df_xtest['data_missing'][timesteps:].to_numpy(), X_hat_test[1:,-1,0]]
# evaluate_label_test = ['rs actual', 'rs predict']
# dp.save_plot(evaluate_data_test, evaluate_label_test, save_dir, 'rs_prediction_test')

# #evaluate y
# evaluate_data_train  = [df_ytrain['data_missing'][timesteps:].to_numpy(), Y_hat_train[1:,-1,0]]
# evaluate_label_train = ['sg actual', 'sg predict']
# dp.save_plot(evaluate_data_train, evaluate_label_train, save_dir, 'sg_prediction_train')

# evaluate_data_test  = [df_ytest['data_missing'][timesteps:].to_numpy(), Y_hat_test[1:,-1,0]]
# evaluate_label_test = ['sg actual', 'sg predict']
# dp.save_plot(evaluate_data_test, evaluate_label_test, save_dir, 'sg_prediction_test')

# # beep sound when finished
# for i in range(1):
#     print("\a")
#     time.sleep(1)
import data_processing as dp
import lstm_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os


# set parameters
params = {
    "input_dim": 2,
    "latent_dim": 20,
    "timesteps": 100,
    "train_size": None,
    "epochs": 100,
    "batch_size": 100,
    "missing_rate": 0.0,
    "data_path": "~/Sensor-Glove/src/data_handler/data/0813_1559_ind_sample.csv"#ind_20min_0812_1436.csv"
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

# 関数を呼び出してデータフレームを取得
df_dummy = dp.make_dummy_data(random_factor=0.05, freq=(5, 10), amp=(1, 1), duration=15, step=0.001)
df = df_dummy
df['rs4'] = df_dummy['X']
df['sg1'] = df_dummy['Y']
df['sg1_filtered'] = df['sg1']
df['rs4_filtered'] = df['rs4']

# df = dp.get_data(file_path)
# df['rs4_filtered'] = df['rs4'].apply(lambda x: x if -90 <= x <= 150 else np.nan)
# df['rs4_lowpass'] = dp.fft_lowpass(df['rs4_filtered'], 5)
# df['sg1_filtered'] = df['sg1'].apply(lambda x: x if 100 <= x <= 600 else np.nan)

# scale data
std_scaler = StandardScaler()
std_scaler.fit(df)
df_std = pd.DataFrame(std_scaler.transform(df), columns=df.columns)


# split data
# df_xtrain, df_xtest = dp.split_data(df_std['t'], df_std['rs4_lowpass'], test_size=0.1)
df_xtrain, df_xtest = dp.split_data(df_std['t'], df_std['rs4_filtered'], test_size=0.1)
df_ytrain, df_ytest = dp.split_data(df_std['t'], df_std['sg1_filtered'], test_size=0.1)
# df_xtrain, df_xtest = dp.split_data(df_std['t'], df_std['rs4'], test_size=0.1)
# df_ytrain, df_ytest = dp.split_data(df_std['t'], df_std['sg1'], test_size=0.1)

# introduce missingness
df_xtrain['data_missing'] = dp.introduce_missingness(df_xtrain['data'], missing_rate)
df_xtrain['if_missing']   = df_xtrain['data_missing'].isnull() # label missing data
df_xtest ['data_missing'] = dp.introduce_missingness(df_xtest['data'],  missing_rate)

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

mask_train = X_train_zero[:,-1,0] != 0
mask_test  = X_test_zero [:,-1,0] != 0

XY_true_zero = np.concatenate([X_train_zero, Y_train_zero], axis=2)

X_train_zero_filtered = X_train_zero[mask_train]
X_test_zero_filtered  = X_test_zero [mask_test]
mask_train = Y_train_zero[:,-1,0] != 0
mask_test  = Y_test_zero [:,-1,0] != 0
Y_train_zero_filtered = Y_train_zero[mask_train]
Y_test_zero_filtered  = Y_test_zero [mask_test]

# get last data
# X_train_last = XY_train_zero[:, -1, 0]
# X_test_last  = XY_test_zero[:, -1, 0]
# XY_train_last = XY_train_zero[:, -1, :]
# XY_test_last  = XY_test_zero[:, -1, :]
# X_test_filtered_last  = XY_test_zero_filtered[:, -1, 0]
# X_train_filtered_last = XY_train_zero_filtered[:, -1, 0]

# make model
# model=lstm_model.lstm_ae(input_dim, latent_dim, timesteps)
model=lstm_model.diff_lstm_ae(input_dim, latent_dim, timesteps)
model.summary()
model.compile(optimizer='adam', loss=lstm_model.loss_missingvalue)
history = model.fit([X_train_zero, Y_train_zero], XY_true_zero, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=([X_train_zero, Y_train_zero], XY_true_zero))


# predict with model
XY_hat_train = model.predict([X_train_zero,Y_train_zero])
XY_hat_test  = model.predict([X_test_zero, Y_test_zero])
print("XY_hat_train.shape : ", XY_hat_train.shape)

# save
params['train_size'] = X_train_zero.shape[0]
dp.save_params(params, 'params')

history_data  = [np.array(history.history['loss']), np.array(history.history['val_loss'])]
history_label = ['Train loss', 'Valid loss']
dp.save_plot(history_data, history_label, 'loss')

train_data  = [df_xtrain['data_missing'].to_numpy(), df_ytrain['data'].to_numpy()]
train_label = ['rs train', 'sg train']   
dp.save_plot(train_data, train_label, 'train_data')

evaluate_data_train  = [df_xtrain['data_missing'][timesteps:].to_numpy(), XY_hat_train[:,-1,0]]
evaluate_label_train = ['rs actual', 'rs predict']
dp.save_plot(evaluate_data_train, evaluate_label_train, 'rs_prediction_train')

evaluate_data_test  = [df_xtest['data_missing'][timesteps:].to_numpy(), XY_hat_test[:,-1,0]]
evaluate_label_test = ['rs actual', 'rs predict']
dp.save_plot(evaluate_data_test, evaluate_label_test, 'rs_prediction_test')

#evaluate y
evaluate_data_train  = [df_ytrain['data'][timesteps:].to_numpy(), XY_hat_train[:,-1,1]]
evaluate_label_train = ['sg actual', 'sg predict']
dp.save_plot(evaluate_data_train, evaluate_label_train, 'sg_prediction_train')

evaluate_data_test  = [df_ytest['data'][timesteps:].to_numpy(), XY_hat_test[:,-1,1]]
evaluate_label_test = ['sg actual', 'sg predict']
dp.save_plot(evaluate_data_test, evaluate_label_test, 'sg_prediction_test')

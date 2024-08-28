import data_processing as dp
import lstm_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import time


# set parameters
params = {
    "input_dim": 1,
    "latent_dim": 10,
    "timesteps": 10,
    "train_size": None,
    "epochs": 100,
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

# data_path2 = "~/Sensor-Glove/src/data_handler/data/ind_20min_0812_1436.csv"
data_path2 = "~/Sensor-Glove/src/data_handler/data/0813_1525_ind_sample.csv"
# data_path2 = "~/Sensor-Glove/src/data_handler/data/0813_{1525,1534,1539,1544,1559}_ind_sample.csv"#ind_0810_1225.csv"
data_path3 = "~/Sensor-Glove/src/data_handler/data/ind_20min_0812_1630.csv"
# get data from csv
file_path = os.path.expanduser(data_path)
file_path2 = os.path.expanduser(data_path2)
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
    
def prep_data(data_path):
    file_path = os.path.expanduser(data_path)
    df = dp.get_data(file_path)
    df['rs4_filtered'] = df['rs4'].apply(lambda x: x if -90 <= x <= 150 else np.nan)
    df['sg1_filtered'] = df['sg1'].apply(lambda x: x if 200 <= x <= 600 else np.nan)
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
    return df_xtrain, df_xtest, df_ytrain, df_ytest, X_train_zero, X_test_zero, Y_train_zero, Y_test_zero

df_xtrain1, df_xtest1, df_ytrain1, df_ytest1, X_train_zero1, X_test_zero1, Y_train_zero1, Y_test_zero1 = prep_data(data_path)
df_xtrain2, df_xtest2, df_ytrain2, df_ytest2, X_train_zero2, X_test_zero2, Y_train_zero2, Y_test_zero2 = prep_data(data_path2)
df_xtrain3, df_xtest3, df_ytrain3, df_ytest3, X_train_zero3, X_test_zero3, Y_train_zero3, Y_test_zero3 = prep_data(data_path3)

X_train_zero = np.concatenate([X_train_zero1, X_train_zero2], axis=0)
X_test_zero  = np.concatenate([X_test_zero1,  X_test_zero2],  axis=0)
Y_train_zero = np.concatenate([Y_train_zero1, Y_train_zero2], axis=0)
Y_test_zero  = np.concatenate([Y_test_zero1,  Y_test_zero2],  axis=0)

X_train_zero = np.concatenate([X_train_zero, X_train_zero3], axis=0)
X_test_zero  = np.concatenate([X_test_zero,  X_test_zero3],  axis=0)
Y_train_zero = np.concatenate([Y_train_zero, Y_train_zero3], axis=0)
Y_test_zero  = np.concatenate([Y_test_zero,  Y_test_zero3],  axis=0)

print(f"X_train_zero.shape : {X_train_zero.shape}, Y_train_zero.shape : {Y_train_zero.shape}")

# make model
# model=lstm_model.lstm_ae(input_dim, latent_dim, timesteps)
model=lstm_model.lstm_ae_2i2o(input_dim, latent_dim, timesteps)
model.summary()
model.compile(optimizer='adam', loss=lstm_model.masked_mse_loss)
history = model.fit([X_train_zero, Y_train_zero], [X_train_zero, Y_train_zero], epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=([X_train_zero, Y_train_zero], [X_train_zero, Y_train_zero]))

# predict with model
X_hat_train, Y_hat_train = model.predict([X_train_zero1,Y_train_zero1])
X_hat_test,  Y_hat_test  = model.predict([X_test_zero1, Y_test_zero1])
# print("XY_hat_train.shape : ", XY_hat_train.shape)

# save
params['train_size'] = X_train_zero.shape[0]
dp.save_params(params, 'params')
dp.save_model(model, 'lstm_ae_2i2o')

history_data  = [np.array(history.history['loss']), np.array(history.history['val_loss'])]
history_label = ['Train loss', 'Valid loss']
dp.save_plot(history_data, history_label, 'loss')

train_data  = [df_xtrain1['data_missing'].to_numpy(), df_ytrain1['data'].to_numpy()]
train_label = ['rs train', 'sg train']   
dp.save_plot(train_data, train_label, 'train_data')

evaluate_data_train  = [df_xtrain1['data_missing'][timesteps:].to_numpy(), X_hat_train[1:,-1,0]]
evaluate_label_train = ['rs actual', 'rs predict']
dp.save_plot(evaluate_data_train, evaluate_label_train, 'rs_prediction_train')

evaluate_data_test  = [df_xtest1['data_missing'][timesteps:].to_numpy(), X_hat_test[1:,-1,0]]
evaluate_label_test = ['rs actual', 'rs predict']
dp.save_plot(evaluate_data_test, evaluate_label_test, 'rs_prediction_test')

#evaluate y
evaluate_data_train  = [df_ytrain1['data'][timesteps:].to_numpy(), Y_hat_train[1:,-1,0]]
evaluate_label_train = ['sg actual', 'sg predict']
dp.save_plot(evaluate_data_train, evaluate_label_train, 'sg_prediction_train')

evaluate_data_test  = [df_ytest1['data'][timesteps:].to_numpy(), Y_hat_test[1:,-1,0]]
evaluate_label_test = ['sg actual', 'sg predict']
dp.save_plot(evaluate_data_test, evaluate_label_test, 'sg_prediction_test')

# beep sound when finished
for i in range(1):
    print("\a")
    time.sleep(1)
import data_processor as dp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lstm_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import time
from tensorflow.keras.models import load_model


file_path = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/results/0817/1156")
# get params
param_path = os.path.join(file_path, "params.json")
params = dp.read_json(param_path)
timesteps = params["timesteps"]

# get model
model_path = os.path.join(file_path, "lstm_ae_2i2o.keras")
model = load_model(model_path, compile=False)
params['model_path'] = model_path

# get data from csv
eval_data_path = "~/Sensor-Glove/src/data_handler/data/0813_1544_ind_sample.csv"#ind_20min_0812_1630.csv"#
file_path = os.path.expanduser(eval_data_path)

df = dp.get_data(file_path)
df['rs4_filtered'] = df['rs4'].apply(lambda x: x if -90 <= x <= 150 else np.nan)
# df['rs4_lowpass'] = dp.fft_lowpass(df['rs4_filtered'], 5)
df['sg1_filtered'] = df['sg1'].apply(lambda x: x if 200 <= x <= 600 else np.nan)

# scale data
std_scaler = StandardScaler()
std_scaler.fit(df)
df_std = pd.DataFrame(std_scaler.transform(df), columns=df.columns)


# split data
df_x_eval, _ = dp.split_data(df_std['t'], df_std['rs4_filtered'], test_size=0.2)
df_y_eval, _ = dp.split_data(df_std['t'], df_std['sg1_filtered'], test_size=0.2)

# introduce missingness
missing_rate = 0.3
df_x_eval['data_missing'] = dp.introduce_missingness(df_x_eval['data'], missing_rate)
df_x_eval['if_missing']   = df_x_eval['data_missing'].isnull() # label missing data
# df_xtest ['data_missing'] = dp.introduce_missingness(df_xtest['data'],  missing_rate)

# make sequence for LSTM
X_eval = dp.make_sequence(df_x_eval['data_missing'], timesteps)
# X_test  = dp.make_sequence(df_xtest['data_missing'],  timesteps)
Y_eval = dp.make_sequence(df_y_eval['data'], timesteps)
# Y_test  = dp.make_sequence(df_ytest['data'],  timesteps)
# 次元を追加
X_eval = np.expand_dims(X_eval, axis=-1)
# X_test  = np.expand_dims(X_test,  axis=-1)
Y_eval = np.expand_dims(Y_eval, axis=-1)
# Y_test  = np.expand_dims(Y_test,  axis=-1)
print(f"X_eval.shape : {X_eval.shape}, Y_eval.shape : {Y_eval.shape}")  

# concatenate X and Y (input dimention of LSTM will be 2)
X_eval_zero = np.nan_to_num(X_eval, nan=0)
# X_test_zero  = np.nan_to_num(X_test , nan=0)
Y_eval_zero = np.nan_to_num(Y_eval, nan=0)
# Y_test_zero  = np.nan_to_num(Y_test , nan=0)


# predict with model
X_hat_eval, Y_hat_eval = model.predict([X_eval_zero,Y_eval_zero])
# X_hat_test,  Y_hat_test  = model.predict([X_test_zero, Y_test_zero])
# print("XY_hat_eval.shape : ", XY_hat_eval.shape)

# save
save_dir = dp.prep_save_dir()
params['eval_size'] = X_eval_zero.shape[0]
dp.save_params(params, save_dir, 'params')
# dp.save_model(model, 'lstm_ae_2i2o')

train_data  = [df_x_eval['data_missing'].to_numpy(), df_y_eval['data'].to_numpy()]
train_label = ['rs input', 'sg input']   
dp.save_plot(train_data, train_label,  save_dir,'input_data')

evaluate_data_train  = [df_x_eval['data_missing'][timesteps:].to_numpy(), X_hat_eval[1:,-1,0]]
evaluate_label_train = ['rs missing', 'rs predict']
dp.save_plot(evaluate_data_train, evaluate_label_train,  save_dir,'rs_missing_prediction')

evaluate_data_train  = [df_x_eval['data'][timesteps:].to_numpy(), X_hat_eval[1:,-1,0]]
evaluate_label_train = ['rs actual', 'rs predict']
dp.save_plot(evaluate_data_train, evaluate_label_train,  save_dir,'rs_prediction')


#evaluate y
evaluate_data_train  = [df_y_eval['data'][timesteps:].to_numpy(), Y_hat_eval[1:,-1,0]]
evaluate_label_train = ['sg actual', 'sg predict']
dp.save_plot(evaluate_data_train, evaluate_label_train,  save_dir,'sg_prediction')


# # beep sound when finished
# for i in range(1):
#     print("\a")
#     time.sleep(1)

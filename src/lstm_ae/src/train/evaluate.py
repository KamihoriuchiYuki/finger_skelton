import data_processor as dp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lstm_model
from sklearn.preprocessing import StandardScaler
import os
import pickle
import data_handler as dh
from tensorflow.keras.models import load_model

# 設定
file_path = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/results/0823/1948")

# data_path = "~/Sensor-Glove/src/data_handler/data/Kanuhoriuchi/2_0816_1829_ind_20min.csv" # 12000, sg = 550 ~ 700
# data_path = "~/Sensor-Glove/src/data_handler/data/Kanuhoriuchi/3_0816_1853_ind_20min.csv" # 12000, sg = 500 ~ 650
# data_path =  "~/Sensor-Glove/src/data_handler/data/Kanuhoriuchi/0816_1739_ind_20min.csv"  # 14000, sg = 350 ~ 450

# data_path = "~/Sensor-Glove/src/data_handler/data/0817_1823_10min_side.csv" # 7000, sg = 500 ~ 620
# data_path = "~/Sensor-Glove/src/data_handler/data/0813_1525_ind_sample.csv" #  350, sg = 450 ~ 550
# data_path = "~/Sensor-Glove/src/data_handler/data/0813_1539_ind_sample.csv" #  600, sg = 375 ~ 475
# data_path = "~/Sensor-Glove/src/data_handler/data/ind_20min_0812_1630.csv"  #12000, sg = 350 ~ 450
# data_path = "~/Sensor-Glove/src/data_handler/data/ind_0810_1225.csv"        # 1200, sg = 300 ~ 400
# data_path = "~/Sensor-Glove/src/data_handler/data/0813_1559_ind_sample.csv" # 6000, sg = 220 ~ 350

# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_220_350_1.csv" 
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_220_350_2.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_300_400.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_1.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_2.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_4.csv"
data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_5.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_6.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_7.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_8.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_9.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_10.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_11.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_2.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_6.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_7.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_9.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_10.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_11.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_12.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_14.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_450_550.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_620_2.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_620_4.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_2.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_5.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_6.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_8.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_10.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_11.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_12.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_1.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_2.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_3.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_4.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_5.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_6.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_7.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_8.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_9.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_10.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_11.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_12.csv"


eval_data_path = os.path.expanduser(data_path)

# パラメータの読み込み
param_path = os.path.join(file_path, "params.json")
params = dp.read_json(param_path)
timesteps = params["timesteps"]
x_missing_rate = 0.4
y_missing_rate = 0.0

# モデルの読み込み
model_path = os.path.join(file_path, "lstm_ae_2i2o.keras")
# model_path = os.path.join(file_path, "autoencoder.keras")
model = load_model(model_path, compile=False)
params['model_path'] = model_path
params['x_missing_rate'] = x_missing_rate
params['y_missing_rate'] = y_missing_rate
params['eval_data_path'] = eval_data_path

# スケーラーの読み込み
# std_scaler_rs4 = pickle.load(open(os.path.join(file_path, "std_scaler_rs4.pkl"), 'rb'))
# std_scaler_sg1 = pickle.load(open(os.path.join(file_path, "std_scaler_sg1.pkl"), 'rb'))
std_scaler_rs4 = None
std_scaler_sg1 = None
# DataHandler を使用してデータを処理
data_handler = dh.DataHandler(eval_data_path, scaler_rs4=std_scaler_rs4, scaler_sg1=std_scaler_sg1)
# data_handler = dh.DataHandler(eval_data_path)
std_scaler_rs4, std_scaler_sg1 = data_handler.get_scalers()
# データの分割と欠損の導入
data_handler.split_data(test_size=0.5)
data_handler.introduce_missingness(x_missing_rate=x_missing_rate, y_missing_rate=y_missing_rate)

# シーケンスの生成
_, X_eval, _, Y_eval, _, X_eval_missing, _, Y_eval_missing = data_handler.make_sequence(timesteps)

# NaNの値をゼロに置き換え
X_eval_zero = np.nan_to_num(X_eval_missing, nan=0)
Y_eval_zero = np.nan_to_num(Y_eval_missing, nan=0)

# モデルによる予測
X_hat_eval, Y_hat_eval = model.predict([X_eval_zero, Y_eval_zero])



# 保存ディレクトリの作成
save_dir = dp.prep_save_dir()
params['eval_size'] = X_eval_zero.shape[0]
dp.save_params(params, save_dir, 'params')

# 逆スケーリング
X_eval_raw = std_scaler_rs4.inverse_transform(X_eval[:, -1, 0].reshape(-1, 1)).flatten()
X_eval_missing = std_scaler_rs4.inverse_transform(X_eval_missing[:, -1, 0].reshape(-1, 1)).flatten()
X_hat_eval = std_scaler_rs4.inverse_transform(X_hat_eval[:, -1, 0].reshape(-1, 1)).flatten()

Y_eval_raw = std_scaler_sg1.inverse_transform(Y_eval[:, -1, 0].reshape(-1, 1)).flatten()
Y_eval_missing = std_scaler_sg1.inverse_transform(Y_eval_missing[:, -1, 0].reshape(-1, 1)).flatten()
Y_hat_eval = std_scaler_sg1.inverse_transform(Y_hat_eval[:, -1, 0].reshape(-1, 1)).flatten()

# X_eval_raw = X_eval[:, -1, 0]
# X_eval_missing = X_eval_missing[:, -1, 0]
# X_hat_eval = X_hat_eval[:, -1, 0]

# Y_eval_raw = Y_eval[:, -1, 0]
# Y_eval_missing = Y_eval_missing[:, -1, 0]
# Y_hat_eval = Y_hat_eval[:, -1, 0]

# グラフのプロット
eval_data = [X_eval_missing, Y_eval_raw]
eval_label = ['RS Eval (missing)', 'SG Eval (raw)']
dp.save_plot(eval_data, eval_label, save_dir, 'eval_data')

# RS値の評価（欠損 vs 予測）
evaluate_data_eval_rs = [X_eval_missing, X_hat_eval]
evaluate_label_eval_rs = ['RS Actual Eval (missing)', 'RS Predicted Eval']
dp.save_plot(evaluate_data_eval_rs, evaluate_label_eval_rs, save_dir, 'rs_missing_prediction')

# RS値の評価（実際 vs 予測）
evaluate_data_eval_rs = [X_eval_raw, X_hat_eval]
evaluate_label_eval_rs = ['RS Actual Eval (raw)', 'RS Predicted Eval']
dp.save_plot(evaluate_data_eval_rs, evaluate_label_eval_rs, save_dir, 'rs_raw_prediction')

# SG値の評価（実際 vs 予測）
evaluate_data_eval_sg = [Y_eval_raw, Y_hat_eval]
evaluate_label_eval_sg = ['SG Actual Eval', 'SG Predicted Eval']
dp.save_plot(evaluate_data_eval_sg, evaluate_label_eval_sg, save_dir, 'sg_prediction')

# beep sound when finished
# for i in range(1):
#     print("\a")
#     time.sleep(1)

import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# CSVファイルからデータを読み込む
# "p1g1_sg480"
# "p1g2_sg600"
# "p1g3_sg350"  # 1728
# "p1g3_sg500"  # 1751
# "p2g1_sg350",  # 1744
# "p2g2_sg550"  # 1747
# "p2g3_sg500" 

base_path = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/important_data/results")
filename = "p1g1_sg480"
# filename = "p1g2_sg600"
# filename = "p1g3_sg350"
# filename = "p1g3_sg500"
# filename = "p2g1_sg350"
# filename = "p2g2_sg550"
# filename = "p2g3_sg500"


data_path = os.path.join(base_path, filename)
df = pd.read_csv(data_path)

# RS Actual Data列を取得し、欠損値を削除する
rs_actual = df[['RS Actual Data']].dropna()
rs_actual_index = rs_actual.index

# RMSEを計算する関数
def calculate_rmse(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return rmse

# 各RS予測データ列とのRMSEを計算
rmse_results = {}
for col in ['RS Predicted Data - RS', 'RS Predicted Data - RS_missing', 'RS Predicted Data - RS_missing_SG', 'RS Predicted Data - RS_SG_both_missing']:
    # 欠損値を削除してインデックスを共通にする
    rs_predicted = df[['RS Actual Data', col]].dropna()
    common_index = rs_actual_index.intersection(rs_predicted.index)
    
    if len(common_index) > 0:  # 共通インデックスが存在する場合
        rs_actual_common = rs_actual.loc[common_index]
        rs_predicted_common = rs_predicted.loc[common_index, col]
        rmse = calculate_rmse(rs_actual_common, rs_predicted_common)
        rmse_results[col] = rmse
    else:
        rmse_results[col] = float('nan')  # 共通インデックスがない場合はNaN

# 結果を表示
for key, value in rmse_results.items():
    if np.isnan(value):
        print(f"RMSE between 'RS Actual Data' and '{key}': No common data")
    else:
        print(f"RMSE between 'RS Actual Data' and '{key}': {value:.4f}")

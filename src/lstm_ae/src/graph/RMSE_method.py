import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import csv
import os

# CSVファイルからデータを読み込む
# data_list = [
#     "p1g3_220_350_1", "p1g3_220_350_2", "p1g3_300_400", "p1g1_300_500_1", "p1g1_300_500_2", 
#     "p1g1_300_500_3", "p1g1_300_500_4", "p1g1_300_500_5", "p1g3_350_440_1", "p1g3_350_440_2", 
#     "p1g3_350_440_4", "p1g3_350_440_5", "p1g3_350_440_6", "p1g3_350_440_7", "p1g3_350_440_8", 
#     "p1g3_350_440_9", "p1g3_350_440_10", "p1g3_350_440_11", "p2g1_350_450_2", "p2g1_350_450_6", 
#     "p2g1_350_450_9", "p2g1_350_450_10", "p2g1_350_450_11", "p2g1_350_450_12", "p2g1_350_450_14", 
#     "p1g3_450_550", "p1g2_450_600_1", "p1g2_450_600_2", "p1g2_450_600_3", "p1g2_450_600_4", 
#     "p1g2_450_600_5", "p1g1_480_570_1", "p1g1_480_570_2", "p1g3_500_620_2", "p1g3_500_620_4", 
#     "p2g3_500_650_2", "p2g3_500_650_5", "p2g3_500_650_6", "p2g3_500_650_8", "p2g3_500_650_10", 
#     "p2g3_500_650_11", "p2g3_500_650_12", "p2g2_550_700_1", "p2g2_550_700_2", "p2g2_550_700_3", 
#     "p2g2_550_700_4", "p2g2_550_700_5", "p2g2_550_700_6", "p2g2_550_700_7", "p2g2_550_700_8", 
#     "p2g2_550_700_10", "p2g2_550_700_11", "p2g2_550_700_12", "p1g2_600_680_2", "p1g2_600_680_3", 
#     "p1g2_600_680_4"
# ]

data_list = [
    "p1g1_300_500_1", #best
    "p1g2_450_600_1", #best
    "p1g3_350_440_9", #best
    "p2g1_350_450_12", # best
    "p2g2_550_700_1", #best
    "p2g3_500_650_12" #best
]

save_path = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/important_data/data_for_paper")
rmse_file_path = os.path.join(save_path, "rmse_results.csv")

# Create or open the CSV file to write the results
with open(rmse_file_path, 'w', newline='') as rf:
    writer = csv.writer(rf)
    rmse_title_list = ['RS Predicted Data - RS', 'RS Predicted Data - RS_missing', 'RS Predicted Data - RS_missing_SG']
    title_list = ['', rmse_title_list[0], rmse_title_list[1], rmse_title_list[2]]
    writer.writerow(title_list)

    # RMSEを計算する関数
    def calculate_rmse(actual, predicted):
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        return rmse

    # Iterate over each data file and calculate RMSE
    for data_name in data_list:
        df = pd.read_csv(os.path.join(save_path.replace("data_for_paper", "results"), f'{data_name}.csv'))

        # RS Actual Data列を取得し、欠損値を削除する
        rs_actual = df[['RS Actual Data']].dropna()
        rs_actual_index = rs_actual.index

        # 各RS予測データ列とのRMSEを計算
        rmse_results = {}
        for col in rmse_title_list:
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

        rmse_data_list = [data_name]

        # 結果を表示し、CSVファイルに書き込み
        for key, value in rmse_results.items():
            if np.isnan(value):
                rmse_data_list.append("")
                print(f"RMSE between 'RS Actual Data' and '{key}': No common data")
            else:
                rmse_data_list.append(value)
                print(f"RMSE between 'RS Actual Data' and '{key}': {value:.4f}")
        writer.writerow(rmse_data_list)

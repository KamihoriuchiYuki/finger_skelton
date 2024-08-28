import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# データファイルのパスリストを定義
# data_paths = ["p1g1_sg480", "p1g2_sg600", "p1g3_sg350", "p1g3_sg500", "p2g1_sg350", "p2g2_sg550", "p2g3_sg500"]
data_paths = ["p1g2_sg600", "p1g3_sg350", "p1g3_sg500", "p2g1_sg350"]

base_path = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/important_data/results")

# プロットの初期設定
fig = plt.figure(figsize=[10, 10])

# StandardScalerのインスタンスを作成
# scaler = StandardScaler()
scaler = MinMaxScaler()

# 各ファイルからデータを読み込んでプロット
for data_path in data_paths:
    # CSVファイルを読み込む
    df = pd.read_csv(os.path.join(base_path, data_path + ".csv"))
    
    # 必要な列を取得し、標準化
    rs_data = df[['RS Actual Data']].values
    sg_data = df[['SG Actual Data']].values
    
    rs_scaled = scaler.fit_transform(rs_data)
    sg_scaled = scaler.fit_transform(sg_data)
    
    # 標準化されたデータをプロット
    plt.scatter(rs_scaled, sg_scaled, marker='o', label=data_path)

# プロットのラベルとタイトルを設定
plt.xlabel('RS Actual Data (Standardized)')
plt.ylabel('SG Actual Data (Standardized)')
plt.title('RS Actual Data vs SG Actual Data (Standardized) for Multiple Files')
plt.legend(loc='upper left')

# グラフを画像ファイルとして保存
fig.savefig('RS_SG_Actual_Data_Multiple_Files_Standardized.png')

# プロットを表示（必要に応じて）
plt.show()

import os
import pandas as pd
import numpy as np

# 距離を計算する関数
def calculate_distance(df):
    df['MCP_Distance'] = np.sqrt(df['MCP_RX']**2 + df['MCP_RY']**2 + df['MCP_RZ']**2)
    df['PIP_Distance'] = np.sqrt(df['PIP_RX']**2 + df['PIP_RY']**2 + df['PIP_RZ']**2)
    df['DIP_Distance'] = np.sqrt(df['DIP_RX']**2 + df['DIP_RY']**2 + df['DIP_RZ']**2)
    df['TIP_Distance'] = np.sqrt(df['TIP_RX']**2 + df['TIP_RY']**2 + df['TIP_RZ']**2)
    return df

# フォルダとファイルのパスを設定
input_folder = 'learning_data_2'
output_folder = 'learning_dist'
os.makedirs(output_folder, exist_ok=True)

# ファイルを処理する
for file_name in os.listdir(input_folder):
    if file_name.startswith('combined_data_') and file_name.endswith('.csv'):
        # CSVファイルを読み込む
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(file_path)
        
        # 距離を計算
        df = calculate_distance(df)
        
        # 新しいファイル名を作成
        new_file_name = file_name.replace('combined_data_', 'combined_dist_')
        new_file_path = os.path.join(output_folder, new_file_name)
        
        # 新しいファイルを保存
        df.to_csv(new_file_path, index=False)

print("すべてのファイルが処理され、保存されました。")

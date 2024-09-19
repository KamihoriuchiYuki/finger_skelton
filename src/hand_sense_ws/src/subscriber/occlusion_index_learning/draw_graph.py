import pandas as pd
import matplotlib.pyplot as plt
import os

# データを読み込み、グラフを作成する関数
def plot_finger_data(file_name, y_min=None, y_max=None):
    # CSVファイルを読み込む
    df = pd.read_csv(file_name)
    
    # グラフを作成
    plt.figure(figsize=(20, 5))
    plt.plot(df['time[s]'], df['MCP'], color="red", label='MCP', linewidth=1)
    plt.plot(df['time[s]'], df['PIP'], color="green", label='PIP', linewidth=1)
    plt.plot(df['time[s]'], df['DIP'], color="blue", label='DIP', linewidth=1)
    plt.plot(df['time[s]'], df['TIP'], color="purple", label='TIP', linewidth=1)
    
    # ラベルとタイトルの設定
    plt.xlabel('time [s]')
    plt.ylabel('distance[mm]')
    plt.title("Distance of Index Finger Joints from Camera")
    plt.legend()
    
    # 縦軸の範囲を指定
    if y_min is not None and y_max is not None:
        plt.ylim([y_min, y_max])
    
    # 画像を保存
    output_file_name = file_name.replace('index_finger_data_', 'index_finger_distance_').replace('.csv', '.png')
    plt.savefig(output_file_name)
    plt.close()
    print(f"グラフが{output_file_name}に保存されました。")

# 使用例
file_name = 'data_index_dist/index_finger_data_20240919_193849.csv'
plot_finger_data(file_name, y_min=100, y_max=300)

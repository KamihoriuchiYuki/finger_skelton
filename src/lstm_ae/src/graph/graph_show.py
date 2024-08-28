import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# CSVファイルからデータを読み込む
data_list = [
    "p1g1_300_500_1",  # best
    "p1g2_450_600_1",  # best
    "p1g3_350_440_9",  # best
    "p2g1_350_450_12",  # best
    "p2g2_550_700_1",  # best
    "p2g3_500_650_12"  # best
]
base_data_path = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/important_data/results/")
save_path = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/important_data/data_for_paper")
# データ範囲の指定
x_range = np.arange(0, 200)

# Loop through each data file in the list
for data_name in data_list:
    # Load the CSV file
    data_path = os.path.join(base_data_path, f"{data_name}.csv")
    df = pd.read_csv(data_path)

    # グラフの設定
    fig, ax1 = plt.subplots(figsize=(25, 8))

    # RSデータを左側の軸でプロット
    ax1.set_xlabel('Row Index')
    ax1.set_ylabel('RS Values', color='tab:red')
    lines1 = []

    line, = ax1.plot(x_range, df['RS Actual Data'].values[x_range], linestyle='--', label='RS Actual Data', color='black')
    lines1.append(line)
    line, = ax1.plot(x_range, df['RS Missing Data'].values[x_range], label='RS Missing Data', color='black')
    lines1.append(line)
    line, = ax1.plot(x_range, df['RS Predicted Data - RS'].values[x_range], label='RS Predicted Data - RS', color='goldenrod')
    lines1.append(line)
    line, = ax1.plot(x_range, df['RS Predicted Data - RS_missing'].values[x_range], label='RS Predicted Data - RS_missing', color='darkturquoise')
    lines1.append(line)
    line, = ax1.plot(x_range, df['RS Predicted Data - RS_missing_SG'].values[x_range], label='RS Predicted Data - RS_missing_SG', color='fuchsia')
    lines1.append(line)

    ax1.tick_params(axis='y', labelcolor='tab:red')

    # SGデータを右側の軸でプロット
    ax2 = ax1.twinx()
    ax2.set_ylabel('SG Values', color='tab:gray')
    lines2 = []

    line, = ax2.plot(x_range, df['SG Actual Data'].values[x_range], label='SG Actual Data', color='silver')
    lines2.append(line)
    # line, = ax2.plot(x_range, df['SG Predicted Data - RS'].values[x_range], label='SG Predicted Data - RS', color='firebrick')
    # lines2.append(line)
    # line, = ax2.plot(x_range, df['SG Predicted Data - RS_missing'].values[x_range], label='SG Predicted Data - RS_missing', color='crimson')
    # lines2.append(line)
    # line, = ax2.plot(x_range, df['SG Predicted Data - RS_missing_SG'].values[x_range], label='SG Predicted Data - RS_missing_SG', color='darkred')
    # lines2.append(line)

    ax2.tick_params(axis='y', labelcolor='tab:gray')

    # 凡例を表示
    lines = lines1 + lines2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # グラフのタイトル
    plt.title(f'RS and SG Data - {data_name}')

    # グラフを画像ファイルとして保存
    plot_save_path = os.path.join(save_path, f'rs_sg_plot_{data_name}.png')
    plt.savefig(plot_save_path, bbox_inches='tight')

    # プロットの表示
    fig.tight_layout()
    # plt.show()

    # Close the figure to avoid memory issues
    plt.close(fig)

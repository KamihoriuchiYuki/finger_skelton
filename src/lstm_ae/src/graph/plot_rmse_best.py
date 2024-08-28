import pandas as pd
import matplotlib.pyplot as plt
import os

# CSVファイルのパスを指定
save_path = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/important_data/data_for_paper")
rmse_file_path = os.path.join(save_path, "rmse_results_best.csv")
# rmse_file_path = os.path.join(save_path, "rmse_results_all.csv")

rmse_title_list = ['RS Predicted Data - RS', 'RS Predicted Data - RS_missing', 'RS Predicted Data - RS_missing_SG']

# CSVファイルからデータを読み込む
df = pd.read_csv(rmse_file_path)

# プロットの設定
plt.figure(figsize=(18, 10))

# RMSEデータ全体を一つの線としてプロット
for col in rmse_title_list:
    plt.plot(df.iloc[:, 0].values, df[col].values, marker='o', label=col)

# y = 10 の線を追加
plt.axhline(y=10, color='r', linestyle='--', label='y = 10')

# x軸のラベルを回転させて見やすくする
plt.xticks(rotation=90)

# プロットのラベルとタイトルを設定
plt.xlabel('Data Files')
plt.ylabel('RMSE')
plt.title('RMSE of Predicted Data vs. Actual Data for Different Models')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# グラフを画像ファイルとして保存
plot_save_path = os.path.join(save_path, 'rmse_plot_combined_with_y10.png')
plt.savefig(plot_save_path, bbox_inches='tight')

# グラフを表示
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import os

# CSVファイルのパスを指定
save_path = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/important_data/data_for_paper")
rmse_file_path = os.path.join(save_path, "rmse_results_best.csv")
# rmse_file_path = os.path.join(save_path, "rmse_results_all.csv")


# CSVファイルからデータを読み込む
df = pd.read_csv(rmse_file_path)

# グループのプレフィックスをリストで定義
group_prefixes = ["p1g1", "p1g2", "p1g3", "p2g1", "p2g2", "p2g3"]
rmse_title_list = ['RS Predicted Data - RS', 'RS Predicted Data - RS_missing', 'RS Predicted Data - RS_missing_SG']

# プロットの設定
plt.figure(figsize=(18, 10))

# 各グループのプロットを作成
for group_prefix in group_prefixes:
    # グループに該当する行をフィルタリング
    group_df = df[df.iloc[:, 0].str.startswith(group_prefix)]
    
    # モデルごとに異なる色とマーカーでRMSEをプロット
    for col in rmse_title_list:
        plt.plot(group_df.iloc[:, 0].values, group_df[col].values, marker='o', label=f'{col} ({group_prefix})')

# y = 10 の線を追加
plt.axhline(y=10, color='r', linestyle='--', label='y = 10')

# x軸のラベルを回転させて見やすくする
plt.xticks(rotation=90)

# プロットのラベルとタイトルを設定
plt.xlabel('Data Files')
plt.ylabel('RMSE')
plt.title('RMSE of Predicted Data vs. Actual Data for Different Models and Groups')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# グラフを画像ファイルとして保存
plot_save_path = os.path.join(save_path, 'rmse_plot_all_groups_with_y10.png')
plt.savefig(plot_save_path, bbox_inches='tight')

# グラフを表示
plt.show()

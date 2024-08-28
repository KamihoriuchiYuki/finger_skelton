import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_processor as dp
from sklearn.preprocessing import StandardScaler


def find_latest_file(directory):
    # フォルダ内の全ファイルを取得
    files = os.listdir(directory)

    # ファイルのフルパスを取得
    full_paths = [os.path.join(directory, f) for f in files]

    # 最新のファイルを取得
    latest_file = max(full_paths, key=os.path.getmtime)


    return latest_file

# 使用例
directory = "~/Sensor-Glove/src/data_handler/data/0813_1631_ind_sample.csv"  # フォルダのパスを指定
directory = os.path.expanduser(directory)  # '~' を展開



# Get file path
# file_path = find_latest_file(directory) # get latest file

# kamihoriuchi
# data_path = "~/Sensor-Glove/src/data_handler/data/Kanuhoriuchi/2_0816_1829_ind_20min.csv" # 12000, sg = 550 ~ 700
# data_path = "~/Sensor-Glove/src/data_handler/data/Kanuhoriuchi/3_0816_1853_ind_20min.csv" # 12000, sg = 500 ~ 650
# data_path =  "~/Sensor-Glove/src/data_handler/data/Kanuhoriuchi/0816_1739_ind_20min.csv"  # 14000, sg = 350 ~ 450

# data_path = "~/Sensor-Glove/src/data_handler/data/0817_1823_10min_side.csv" # 7000, sg = 500 ~ 620
# data_path = "~/Sensor-Glove/src/data_handler/data/0813_1525_ind_sample.csv" #  350, sg = 450 ~ 550
# data_path = "~/Sensor-Glove/src/data_handler/data/0813_1539_ind_sample.csv" #  600, sg = 375 ~ 475
data_path = "~/Sensor-Glove/src/data_handler/data/ind_20min_0812_1630.csv"  #12000, sg = 350 ~ 440
# data_path = "~/Sensor-Glove/src/data_handler/data/ind_0810_1225.csv"        # 1200, sg = 300 ~ 400
# data_path ="~/Sensor-Glove/src/data_handler/data/0813_1559_ind_sample.csv" # 6000, sg = 220 ~ 350
# data_path = "~/Sensor-Glove/src/data_handler/data/0813_1534_ind_sample.csv" #  400, sg = 380 ~ 480 kitanai
# data_path = "~/Sensor-Glove/src/data_handler/data/0813_1544_ind_sample.csv" #  400, sg = 350 ~ 450 kitanai

### divided data
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_220_350_1.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_220_350_2.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_300_400.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_1.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_2.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_3.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_4.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_5.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_6.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_7.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_8.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_9.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_10.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_11.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_12.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_13.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_14.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_450_550.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_620_2.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_620_4.csv"
data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_2.csv" #kami 3
data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_5.csv" 
data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_6.csv"
data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_8.csv"
data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_10.csv"
data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_11.csv"
data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_12.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_1.csv" #kami2
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


file_path = os.path.expanduser(data_path) # get specific file

# 関数を呼び出してデータフレームを取得
df = dp.get_data(file_path)


# add nan filter 
df['rs5_filtered'] = df['rs5'].apply(lambda x: x if -90 <= x <= 100 else np.nan)
df['rs4_filtered'] = df['rs4'].apply(lambda x: x if -90 <= x <= 140 else np.nan)
df['sg1_filtered'] = df['sg1'].apply(lambda x: x if 200 <= x <= 800 else np.nan)
# df['sg1_filtered'] = df['sg1'].apply(lambda x: x if 470 <= x <= 570 else np.nan)

# lowpass filter
df['rs4_lowpass'] = dp.fft_lowpass(df['rs4'], 5)
df['sg1_lowpass'] = dp.fft_lowpass(df['sg1_filtered'], 5)

# scale data
std_scaler = StandardScaler()
std_scaler.fit(df)
df_std = pd.DataFrame(std_scaler.transform(df), columns=df.columns)
# df_inverse = pd.DataFrame(std_scaler.inverse_transform(df_std), columns=df.columns)

std_scaler_rs4 = StandardScaler()
std_scaler_rs4.fit(df['rs4_filtered'].values.reshape(-1, 1))
df['rs4_scaled'] = std_scaler_rs4.fit_transform(df[['rs4_filtered']])

std_scaler_sg1 = StandardScaler()
std_scaler_sg1.fit(df['sg1_filtered'].values.reshape(-1, 1))
df['sg1_scaled'] = std_scaler_sg1.fit_transform(df[['sg1_filtered']])

# plot
fig = plt.figure(figsize=(25, 5))
fig.suptitle(file_path)
# fig2 = plt.figure(figsize=(25, 5))

# plot sensor glove data
ax = fig.add_subplot(611)
ax2= fig.add_subplot(612)
ax3= fig.add_subplot(613)
ax4= fig.add_subplot(614)
ax5= fig.add_subplot(615)
ax6= fig.add_subplot(616)

# for i, (axis, start, end) in enumerate(zip([ax, ax2, ax3, ax4, ax5, ax6],
#                                            [0, 1000, 2000, 3000, 4000, 5000],
#                                            [1000, 2000, 3000, 4000, 5000, 6000])):
#     axis.plot(df['sg1_filtered'][start:end], label=f'sg1_filtered {start}:{end}')
#     axis.set_xticks(np.arange(start, end, 25))  # x軸のメジャーティックを50単位で設定
#     axis.grid(True, which='both', axis='both')  # グリッドを表 示
#     axis.legend(loc='upper right')  # 凡例を右上に表示

# for i, (axis, start, end) in enumerate(zip([ax, ax2, ax3, ax4, ax5, ax6],
#                                            [0, 1000, 2000, 3000, 4000, 5000],
#                                            [1000, 2000, 3000, 4000, 5000, 6000])):
#     axis.plot(df['rs4_filtered'][start:end], label=f'rs4_filtered {start}:{end}')
#     axis.set_xticks(np.arange(start, end, 25))  # x軸のメジャーティックを50単位で設定
#     axis.grid(True, which='both', axis='both')  # グリッドを表 示
#     axis.legend(loc='upper right')  # 凡例を右上に表示

for i, (axis, start, end) in enumerate(zip([ax, ax2, ax3, ax4, ax5, ax6],
                                           [6000, 7000, 8000, 9000, 10000, 11000],
                                           [7000, 8000, 9000, 10000, 11000, 12000])):
    axis.plot(df['rs4_filtered'][start:end], label=f'rs4_filtered {start}:{end}')
    axis.set_xticks(np.arange(start, end, 25))  # x軸のメジャーティックを50単位で設定
    axis.grid(True, which='both', axis='both')  # グリッドを表 示
    axis.legend(loc='upper right')  # 凡例を右上に表示


# # ax.plot(df['sg1'], label='sg1')
# ax.plot(df['sg1_filtered'][000:1000], label='sg1_filtered')
# ax2.plot(df['sg1_filtered'][1000:2000], label='sg1_filtered')
# ax3.plot(df['sg1_filtered'][2000:3000], label='sg1_filtered')
# ax4.plot(df['sg1_filtered'][3000:4000], label='sg1_filtered')
# ax5.plot(df['sg1_filtered'][4000:5000], label='sg1_filtered')
# ax6.plot(df['sg1_filtered'][5000:6000], label='sg1_filtered')
# ax.plot(df['sg1_filtered'][6000:7000], label='sg1_filtered')
# ax2.plot(df['sg1_filtered'][7000:8000], label='sg1_filtered')
# ax3.plot(df['sg1_filtered'][8000:9000], label='sg1_filtered')
# ax4.plot(df['sg1_filtered'][9000:10000], label='sg1_filtered')
# ax5.plot(df['sg1_filtered'][10000:11000], label='sg1_filtered')
# ax6.plot(df['sg1_filtered'][11000:12000], label='sg1_filtered')

# ax.plot(df['sg1_filtered'][6000:12000], label='sg1_filtered')
# ax.plot(df['sg1_filtered'][12000:18000], label='sg1_filtered')
# ax.plot(df['sg1_scaled'], label='sg1_scaled')
# ax.plot(df['sg1_lowpass'], label='sg1_lowpass')

# plot real sense data
# ax2.plot(df['rs4_filtered'][800:2000], label='rs4')
# ax2.plot(df['sg1_filtered'][000:1000], label='sg1_filtered')
# ax2.plot(df_std['rs4_filtered'], label='rs4_filtered')
# ax2.plot(df['rs4_scaled'], label='rs4_scaled')
# ax2.plot(df['rs5'], label='rs5')
# ax2.plot(df['rs5_filtered'][:], label='rs5_filtered')
# ax2.plot(df['rs4_lowpass'], label='rs4_lowpass')

# ax.legend (loc = 'upper right')
# ax2.legend(loc = 'upper right')

plt.show()
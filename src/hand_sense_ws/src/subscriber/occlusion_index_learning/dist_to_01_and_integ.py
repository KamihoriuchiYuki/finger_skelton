import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(10.6, 13.8), (18.4, 20.45), (24.2, 27.1), (32.2, 34.05), (35.5, 37.0), (39.05, 41.3), (45.75, 45.95), (50.85, 55.05), (63.8, 70.1), (79.65, 86.65), (94.85, 96.9), (103.75, 108.3)],
    'PIP': [(3.4, 6.3), (9.5, 10.5), (13.3, 14.6), (17.4, 18.45), (20.3, 20.8), (23.2, 24.3), (26.85, 28.2), (30.8, 32.3), (33.9, 34.1), (35.35, 35.5), (36.8, 37.2), (38.9, 39.05), (41.05, 42.15), (42.6, 46.3), (48.8, 51.0), (54.85, 56.9), (62.15, 64.0), (69.6, 72.6), (77.6, 80.35), (86.15, 88.25), (92.65, 95.4), (96.8, 97.1), (102.25, 104.1), (108.0, 110.5)],
    'DIP': [(3.0, 4.35), (5.55, 6.65), (9.35, 10.2), (14.3, 14.65), (17.2, 18.2), (20.5, 20.95), (23.15, 23.8), (27.45, 28.35), (30.6, 31.65), (34.0, 34.2), (35.35, 35.4), (37.0, 37.35), (38.9, 39.0), (41.55, 42.3), (44.4, 45.4), (46.15, 46.3), (48.6, 50.5), (55.7, 57.15), (61.3, 63.45), (70.55, 73.0), (77.2, 79.2), (87.65, 88.65), (91.7, 94.15), (97.0, 97.2), (102.0, 103.5), (109.2, 110.75)],
    'TIP': [(34.1, 34.2)]
}

# time[s]の範囲と値に基づいて変換する関数
def transform_value(value, time_value, intervals):
    # time[s]の範囲に基づく処理
    if any(start <= time_value <= end for start, end in intervals):
        return 1
    # 値の範囲に基づく処理
    elif 100 <= value <= 1000:
        return 0
    else:
        return 1

# time[s]列を除くすべての列に対して処理を行う
columns_to_modify = distance_df.columns[distance_df.columns != 'time[s]']

for col in columns_to_modify:
    intervals = time_intervals.get(col, [])
    distance_df[col] = distance_df.apply(lambda row: transform_value(row[col], row['time[s]'], intervals), axis=1)

# 変換されたdistance_dfとposition_dfをtime[s]列で結合
merged_df = pd.merge(distance_df, position_df, on='time[s]')

# 結果を表示
print(merged_df)

# 必要に応じてCSVファイルに保存
merged_df.to_csv(f'learning_data/combined_data_{oooo}.csv', index=False)

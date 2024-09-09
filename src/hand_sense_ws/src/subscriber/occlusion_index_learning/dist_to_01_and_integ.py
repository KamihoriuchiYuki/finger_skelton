import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(4.3, 8.3), (13.65, 17.0), (22.05, 27.0), (34.15, 38.55), (51.05, 54.5), (60.0, 63.4), (69.1, 72.8)],
    'PIP': [(3.45, 4.0), (8.45, 8.9), (13.0, 13.6), (16.95, 17.7), (21.7, 22.05), (27.1, 28.15), (32.65, 33.95), (38.65, 39.95), (49.8, 50.8), (54.8, 55.45), (58.95, 59.75), (63.55, 64.4), (68.25, 68.9), (73.0, 73.85)],
    'DIP': [(2.9, 3.75), (8.4, 9.1), (12.85, 13.35), (17.15, 18.2), (21.35, 21.9), (27.4, 28.4), (32.25, 33.65), (39.05, 40.15), (49.45, 50.55), (55.0, 55.7), (58.6, 59.5), (63.75, 64.7), (68.05, 68.65), (73.05, 74.1)],
    'TIP': [(0.0, 0.0)]
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

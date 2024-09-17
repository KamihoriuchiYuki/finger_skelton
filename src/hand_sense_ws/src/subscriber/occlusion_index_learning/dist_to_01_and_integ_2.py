import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index_2/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index_2/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(3.9, 8.65), (15.3, 19.1), (28.3, 34.1), (40.6, 45.9), (55.0, 59.1), (64.45, 69.1)],
    'PIP': [(2.8, 4.25), (8.1, 9.8), (14.45, 15.75), (18.7, 20.5), (27.7, 28.8), (33.65, 35.1), (40.0, 41.15), (45.7, 46.85), (54.2, 55.45), (58.75, 59.95), (64.05, 65.0), (68.7, 70.0)],
    'DIP': [(2.4, 3.75), (8.7, 10.6), (13.95, 15.4), (19.2, 21.35), (27.1, 28.45), (34.4, 35.6), (39.5, 41.1), (45.9, 47.3), (53.15, 55.45), (58.9, 60.7), (63.7, 64.8), (69.0, 70.5)],
    'TIP': [(0.3, 0.35), (0.4, 0.6), (0.7, 0.75), (1.3, 1.35), (11.25, 11.3), (11.5, 11.6), (13.55, 13.6), (21.4, 21.75), (22.0, 22.05), (23.7, 23.8), (24.05, 24.15), (27.4, 27.5), (27.55, 27.6)]
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
merged_df.to_csv(f'learning_data_2/combined_data_{oooo}.csv', index=False)

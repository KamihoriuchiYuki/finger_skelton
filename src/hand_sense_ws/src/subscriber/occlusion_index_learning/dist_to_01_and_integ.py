import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(7.6, 13.9), (23.1, 23.65), (24.6, 30.0), (38.4, 42.4), (51.3, 55.85), (65.65, 70.25)],
    'PIP': [(5.35, 8.45), (13.2, 15.6), (21.1, 25.25), (29.7, 32.0), (37.2, 38.55), (42.1, 43.9), (49.7, 51.4), (55.3, 57.35), (64.0, 64.05), (64.2, 65.9), (70.05, 71.2)],
    'DIP': [(5.1, 7.05), (14.4, 16.3), (20.65, 24.1), (30.85, 32.8), (36.7, 38.1), (43.25, 44.2), (49.2, 51.05), (56.0, 57.7), (63.3, 65.25), (70.4, 71.45)],
    'TIP': [(16.0, 16.05), (23.1, 23.65), (57.15, 57.2), (57.6, 57.65)]
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

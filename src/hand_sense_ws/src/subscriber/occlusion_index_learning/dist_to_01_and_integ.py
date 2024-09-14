import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(5.25, 10.2), (18.7, 23.8), (31.95, 35.5), (44.4, 47.8), (56.0, 58.2), (63.5, 66.35), (73.0, 74.8)],
    'PIP': [(3.7, 6.5), (8.9, 12.1), (17.0, 19.8), (22.5, 25.45), (30.6, 32.9), (35.3, 36.95), (42.1, 45.0), (47.55, 49.35), (54.4, 56.4), (58.1, 58.6), (62.4, 64.1), (65.9, 67.7), (71.1, 73.7), (74.4, 75.25)],
    'DIP': [(3.15, 4.9), (10.55, 12.35), (16.6, 18.4), (23.65, 26.1), (30.2, 31.75), (35.5, 37.35), (41.65, 43.95), (48.4, 50.0), (54.1, 55.6), (58.3, 59.15), (62.3, 63.25), (66.6, 68.0), (70.9, 72.8), (74.85, 75.4)],
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

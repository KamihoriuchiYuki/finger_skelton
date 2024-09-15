import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(4.85, 7.2), (8.7, 10.45), (10.55, 10.6), (10.8, 10.85), (10.9, 11.05), (11.4, 12.8), (20.1, 20.2), (20.3, 23.3), (30.05, 31.85), (32.0, 32.7), (33.65, 35.4)],
    'PIP': [(3.5, 5.1), (12.65, 13.9), (18.6, 20.35), (23.25, 25.05), (28.6, 30.25), (35.1, 36.6)],
    'DIP': [(2.8, 4.3), (12.75, 14.45), (17.6, 19.55), (23.5, 25.55), (27.7, 29.75), (35.45, 37.4)],
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

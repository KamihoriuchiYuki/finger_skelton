import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index_2/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index_2/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(5.6, 9.25), (16.7, 20.2), (26.4, 29.95), (36.85, 41.1), (47.6, 51.7)],
    'PIP': [(4.3, 5.8), (8.8, 10.2), (15.5, 17.2), (20.0, 20.75), (25.4, 27.1), (29.7, 30.6), (35.95, 37.8), (40.6, 41.7), (47.0, 48.4), (51.4, 52.2)],
    'DIP': [(3.55, 5.35), (9.7, 10.85), (14.5, 16.45), (20.25, 20.95), (24.45, 26.55), (30.0, 30.9), (35.25, 36.95), (41.1, 42.1), (46.45, 47.75), (51.7, 52.85)],
    'TIP': [(5.85, 6.0)]
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

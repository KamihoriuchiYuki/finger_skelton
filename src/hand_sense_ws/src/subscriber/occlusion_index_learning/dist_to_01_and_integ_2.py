import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index_2/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index_2/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(5.8, 11.25), (20.2, 28.25), (29.5, 32.1), (47.8, 50.8)],
    'PIP': [(4.1, 6.5), (10.1, 12.55), (19.1, 21.1), (31.8, 33.7), (41.95, 44.4), (52.5, 53.6)],
    'DIP': [(2.95, 5.9), (11.15, 13.9), (18.0, 20.4), (32.9, 34.05), (41.85, 43.0), (52.9, 54.0)],
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
merged_df.to_csv(f'learning_data_2/combined_data_{oooo}.csv', index=False)

import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index_dist/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index_dist/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(4.5, 8.35), (16.25, 18.9), (27.1, 30.45)],
    'PIP': [(3.55, 5.15), (7.95, 9.4), (14.8, 16.7), (18.55, 19.7), (25.7, 27.3), (30.4, 31.6)],
    'DIP': [(2.35, 4.4), (8.7, 10.05), (14.1, 15.95), (19.15, 20.2), (24.95, 26.75), (30.8, 32.05)],
    'TIP': [(5.45, 5.5), (28.6, 29.2)]
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
merged_df.to_csv(f'learning_dist/combined_dist_{oooo}.csv', index=False)

import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(6.55, 11.6), (20.3, 25.1), (55.3, 59.1), (82.8, 86.25), (90.15, 93.5)],
    'PIP': [(4.5, 7.1), (11.0, 12.85), (18.6, 21.1), (24.4, 26.35), (26.4, 26.65), (54.3, 56.2), (58.85, 59.85), (63.25, 65.0), (81.9, 83.2), (85.9, 86.75), (89.55, 90.6), (93.1, 93.9), (97.0, 104.75)],
    'DIP': [(3.8, 7.05), (6.2, 16.3), (11.8, 13.2), (18.3, 20.25), (25.25, 27.35), (53.95, 55.5), (59.05, 60.0), (63.1, 65.05), (81.55, 82.6), (86.3, 86.95), (89.45, 90.1), (93.3, 94.25), (96.55, 105.05)],
    'TIP': [(18.05, 18.1), (101.3, 101.35)]
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

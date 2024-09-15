import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(4.3, 7.35), (10.9, 13.85), (17.8, 21.35), (22.65, 27.75), (32.5, 36.6), (40.45, 44.85)],
    'PIP': [(3.15, 5.05), (6.9, 8.0), (10.45, 11.3), (13.65, 14.4), (16.85, 18.95), (21.05, 22.9), (27.15, 28.6), (31.9, 32.95), (35.9, 37.3), (40.1, 40.7), (44.55, 45.45)],
    'DIP': [(2.55, 4.1), (7.35, 8.1), (10.35, 10.85), (13.9, 14.6), (16.6, 17.45), (21.35, 22.5), (27.8, 28.65), (31.8, 32.4), (36.75, 37.45), (39.9, 40.4), (44.85, 45.65)],
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

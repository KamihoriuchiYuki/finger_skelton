import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(5.65, 10.45), (22.5, 26.6), (30.05, 33.35), (37.3, 40.5), (44.4, 47.6), (51.1, 54.3), (57.7, 59.35), (60.3, 61.9), (64.3, 68.55)],
    'PIP': [(4.4, 6.2), (10.1, 12.05), (16.55, 19.35), (22.3, 22.7), (26.5, 27.2), (29.6, 30.3), (33.25, 34.0), (36.85, 37.8), (40.35, 41.45), (44.05, 44.5), (47.35, 48.4), (50.75, 51.45), (53.8, 55.0), (57.3, 58.05), (58.8, 60.9), (61.65, 62.75), (64.1, 64.65), (68.3, 69.25)],
    'DIP': [(3.9, 5.85), (10.3, 12.55), (15.45, 19.6), (22.2, 22.7), (26.6, 27.45), (29.35, 30.0), (33.35, 34.25), (36.6, 37.35), (40.4, 41.65), (43.9, 44.4), (47.7, 48.5), (50.7, 51.0), (54.05, 55.15), (57.2, 57.85), (59.2, 60.3), (61.9, 62.8), (64.05, 64.4), (68.4, 69.3)],
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

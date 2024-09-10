import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(5.95, 11.25), (18.9, 22.75), (26.9, 30.3), (36.15, 36.5), (36.55, 39.6), (44.35, 47.85), (50.9, 52.7), (57.3, 59.6), (63.2, 66.0), (69.45, 71.6)],
    'PIP': [(3.9, 7.1), (10.85, 12.25), (17.5, 20.4), (22.4, 23.6), (26.15, 27.45), (30.2, 30.5), (34.75, 36.75), (38.7, 40.1), (43.6, 44.75), (47.5, 48.4), (50.5, 51.1), (52.6, 52.75), (56.85, 57.8), (59.5, 60.35), (62.45, 63.4), (65.6, 66.7), (68.95, 69.9), (71.4, 72.0)],
    'DIP': [(3.3, 5.65), (11.25, 13.15), (17.05, 18.85), (22.8, 23.8), (25.95, 26.85), (30.3, 30.6), (34.4, 35.6), (39.6, 40.3), (43.25, 44.35), (47.8, 48.55), (50.35, 50.8), (52.7, 52.8), (56.5, 57.25), (59.7, 60.4), (62.4, 63.2), (66.0, 66.8), (68.9, 69.3), (71.6, 72.2)],
    'TIP': [(50.05, 50.2), (56.35, 56.4), (68.8, 68.9)]
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

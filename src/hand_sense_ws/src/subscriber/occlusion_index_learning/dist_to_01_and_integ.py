import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(15.25, 20.1), (28.05, 32.4), (37.85, 44.85), (48.9, 55.1)],
    'PIP': [(3.9, 8.95), (13.8, 15.25), (20.2, 21.7), (26.4, 26.45), (26.6, 28.0), (32.6, 33.55), (37.85, 39.0), (43.5, 44.85), (48.9, 49.0), (49.1, 50.2), (53.8, 55.1)],
    'DIP': [(3.2, 9.15), (13.5, 24.25), (20.45, 21.9), (26.05, 27.85), (32.7, 33.9), (37.45, 38.7), (43.8, 45.1), (48.45, 48.5), (48.55, 49.5), (53.9, 55.3)],
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

import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index_dist/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index_dist/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(5.1, 8.5), (18.1, 23.75), (33.1, 36.75), (46.95, 50.1)],
    'PIP': [(3.7, 5.7), (8.0, 9.6), (16.8, 19.95), (22.6, 24.6), (31.8, 33.85), (35.85, 37.4), (46.05, 47.45), (49.5, 50.7)],
    'DIP': [(2.95, 5.25), (8.45, 10.6), (15.7, 18.2), (23.8, 25.7), (30.9, 33.15), (37.05, 38.0), (45.55, 46.8), (50.35, 51.5)],
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
merged_df.to_csv(f'learning_dist/combined_dist_{oooo}.csv', index=False)

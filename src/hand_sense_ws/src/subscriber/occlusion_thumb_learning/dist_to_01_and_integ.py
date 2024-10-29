import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_dist/thumb_data_{oooo}.csv')
position_df = pd.read_csv(f'data_dist/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(0.0, 0.0)],
    'PIP': [(5.6, 7.35), (11.1, 12.8), (17.1, 19.5)],
    'DIP': [(5.5, 7.45), (10.95, 12.9), (16.7, 19.55)],
    'TIP': [(5.25, 7.85), (10.85, 13.1), (16.55, 19.8)]
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

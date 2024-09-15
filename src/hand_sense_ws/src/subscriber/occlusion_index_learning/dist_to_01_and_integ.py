import pandas as pd

# ユーザーが入力するoooo部分
oooo = input("Enter the identifier (oooo): ")

# CSVファイルをdata_indexフォルダーから読み込む
distance_df = pd.read_csv(f'data_index/index_finger_data_{oooo}.csv')
position_df = pd.read_csv(f'data_index/finger_joint_positions_{oooo}.csv')

# 各列に対するtime[s]の範囲設定
time_intervals = {
    'MCP': [(5.0, 9.0), (16.8, 21.75), (27.8, 31.4), (37.4, 41.55), (47.3, 50.6)],
    'PIP': [(3.35, 5.8), (8.65, 10.8), (15.35, 17.6), (21.1, 23.3), (26.7, 28.35), (31.0, 32.5), (36.6, 37.95), (41.3, 42.75), (46.8, 47.7), (50.3, 51.2)],
    'DIP': [(3.15, 4.8), (9.35, 11.1), (14.9, 16.75), (21.8, 23.5), (26.25, 27.7), (31.35, 32.85), (36.3, 37.4), (41.65, 42.8), (46.5, 47.3), (50.65, 51.45)],
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

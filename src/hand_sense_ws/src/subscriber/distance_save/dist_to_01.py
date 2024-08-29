import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('index_finger_data.csv')

# 各列に対して設定する複数のtime[s]の区間
time_intervals = {
    'MCP': [(4.75, 10.05)],
    'PIP': [(3.9, 4.7), (10.02, 10.4)],
    'DIP': [(0.0, 0.0), (0.0, 0.0)],
    'TIP': [(0.0, 0.0)]
}

# 各列の複数の区間に基づいて値を1または0に変更
for col, intervals in time_intervals.items():
    df[col] = df.apply(
        lambda row: 1 if any(start <= row['time[s]'] <= end for start, end in intervals) else 0, 
        axis=1
    )

# 結果を表示
print(df)

# 必要に応じてCSVファイルに保存
df.to_csv('modified_csv_file.csv', index=False)
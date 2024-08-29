import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('your_csv_file.csv')

# time[s]の範囲と列を指定
time_start = 0.2
time_end = 0.6
columns_to_modify = ['MCP', 'PIP', 'DIP', 'TIP']

# time[s]の範囲内にある要素を1、範囲外の要素を0に変更
for col in columns_to_modify:
    df[col] = df.apply(lambda row: 1 if time_start <= row['time[s]'] <= time_end else 0, axis=1)

# 結果を表示
print(df)

# 必要に応じてCSVファイルに保存
df.to_csv('modified_csv_file.csv', index=False)

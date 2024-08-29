import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('index_finger_data.csv')

# 各列に対する設定
time_intervals = {
    'MCP': [(4.75, 10.05)],
    'PIP': [(3.9, 4.7), (10.02, 10.4)],
    'DIP': [(0.0, 0.0), (0.0, 0.0)],
    'TIP': [(0.0, 0.0)]
}

# 各列の値が100~1000の範囲であれば0、それ以外は1に変更
def transform_value(value, intervals):
    # time[s]の範囲に基づく処理
    if any(start <= value <= end for start, end in intervals):
        return 1
    # 値の範囲に基づく処理
    elif 100 <= value <= 1000:
        return 0
    else:
        return 1

# time[s]列を除くすべての列に対して処理を行う
columns_to_modify = df.columns[df.columns != 'time[s]']

for col in columns_to_modify:
    intervals = time_intervals.get(col, [])
    df[col] = df[col].apply(lambda x: transform_value(x, intervals))

# 結果を表示
print(df)

# 必要に応じてCSVファイルに保存
df.to_csv('modified_csv_file.csv', index=False)
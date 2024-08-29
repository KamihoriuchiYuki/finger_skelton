import pandas as pd
import os

def process_and_merge_csv_files(oooo):
    # ファイルパスを生成
    distance_file_name = f'index_finger_distance_{oooo}.csv'
    position_file_name = f'index_finger_position_{oooo}.csv'
    
    distance_file_path = os.path.join('data_index', distance_file_name)
    position_file_path = os.path.join('data_index', position_file_name)
    
    # CSVファイルを読み込む
    distance_df = pd.read_csv(distance_file_path)
    position_df = pd.read_csv(position_file_path)
    
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
    columns_to_modify = distance_df.columns[distance_df.columns != 'time[s]']

    for col in columns_to_modify:
        intervals = time_intervals.get(col, [])
        distance_df[col] = distance_df[col].apply(lambda x: transform_value(x, intervals))

    # 2つのデータフレームを統合する
    merged_df = pd.merge(distance_df, position_df, on='time[s]')

    # 結果を表示
    print(merged_df)

    # 統合された結果を新しいCSVファイルに保存
    output_file_path = os.path.join('data_index', f'merged_index_finger_data_{oooo}.csv')
    merged_df.to_csv(output_file_path, index=False)
    print(f"Merged file saved as {output_file_path}")

# 例として、ooooに'1234'を指定
process_and_merge_csv_files('1234')

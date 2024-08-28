import pandas as pd

# ターミナルからlの値を入力
l = int(input("lの値を入力してください: "))

# Bファイルのパスと名前を指定 (lの値をファイル名に反映)
<<<<<<< Updated upstream
B_file_path = f'/home/hlab6/Sensor-Glove/src/data_handler/data/deivide/sg_350_440_{l}.csv'
=======
B_file_path = f'/home/hlab6/Sensor-Glove/src/data_handler/data/divided/sg_500_620_{l}.csv'
>>>>>>> Stashed changes

print(f'Bファイルパス: {B_file_path}')  # デバッグ出力

# Bファイルを新しく作成する（既存ファイルを上書きする形で空のファイルを作成）
with open(B_file_path, 'w') as f:
    pass

# Aファイルのパスと名前を指定
A_file_path = '~/Sensor-Glove/src/data_handler/data/ind_20min_0812_1630.csv'

print(f'Aファイルパス: {A_file_path}')  # デバッグ出力

# Aファイルを読み込む
try:
    A_df = pd.read_csv(A_file_path, header=None)
    print('Aファイルの読み込み成功')  # デバッグ出力
except FileNotFoundError as e:
    print(f'エラー: Aファイルが見つかりません - {e}')
    exit()

# Bファイルを新しく作成するために空のデータフレームを初期化
B_df = pd.DataFrame()

# ① Aファイルの0行目をBファイルの0行目に挿入
try:
    B_df = pd.concat([B_df, A_df.iloc[[0]]], ignore_index=True)
    print('0行目の挿入成功')  # デバッグ出力
except Exception as e:
    print(f'エラー: {e}')
    exit()

# ② Aファイルのn-n+k行目をBファイルの1〜1+k行目に挿入
n = 1000 * (l - 1)  # ここでnの値を指定
k = 1000 * l  # ここでkの値を指定
try:
    B_df = pd.concat([B_df, A_df.iloc[n-n:n-n+k+1].reset_index(drop=True)], ignore_index=True)
    print(f'{n}行目から{k+1}行目までの挿入成功')  # デバッグ出力
except Exception as e:
    print(f'エラー: {e}')
    exit()

# 指定されたパスに新しいBファイルを作成して保存する
try:
    B_df.to_csv(B_file_path, header=False, index=False)
    print(f'新しいBファイルが作成されました: {B_file_path}')
except Exception as e:
    print(f'エラー: {e}')

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# フォルダ内のすべてのCSVファイルを読み込む
def load_all_data(folder_path):
    all_X = []
    all_y = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path)
            X, y = prepare_data(data)
            all_X.append(X)
            all_y.append(y)
    
    # 各ファイルごとに処理されたX, yを結合
    return pd.concat(all_X, ignore_index=True), pd.concat(all_y, ignore_index=True)

# データの準備
def prepare_data(data):
    X = data[[
        'Wrist_X', 'Wrist_Y', 'Wrist_Z',
        'MCP_X', 'MCP_Y', 'MCP_Z',
        'PIP_X', 'PIP_Y', 'PIP_Z',
        'DIP_X', 'DIP_Y', 'DIP_Z',
        'TIP_X', 'TIP_Y', 'TIP_Z'
    ]]
    
    # 座標の変化率を追加
    X_diff = X.diff().fillna(0)
    X = pd.concat([X, X_diff], axis=1)
    
    # 1ステップ前の信頼値を追加（ただしファイルの最初の行はゼロで埋める）
    y = data[['MCP', 'PIP', 'DIP', 'TIP']]
    y_prev = y.shift(1).fillna(0)
    
    # 各ファイルの最初の行はゼロに置き換える
    y_prev.iloc[0] = 0
    
    X = pd.concat([X, y_prev], axis=1)
    
    return X, y

# 学習モデルのトレーニング
def train_model(X, y, model_path):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

# モデルのロード
def load_model(model_path):
    return joblib.load(model_path)

# 学習データの準備
folder_path = 'learning_data'
X, y = load_all_data(folder_path)

# モデルのトレーニング
model_path = 'joint_reliability_model_integ.pkl'
model = train_model(X, y, model_path)

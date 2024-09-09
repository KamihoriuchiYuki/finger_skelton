import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# CSVファイルの読み込み
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

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
    
    # 1ステップ前の信頼値を追加
    y = data[['MCP', 'PIP', 'DIP', 'TIP']]
    y_prev = y.shift(1).fillna(0)
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
file_path = 'learning_data/combined_data_20240830_195300.csv'
data = load_data(file_path)
X, y = prepare_data(data)

# モデルのトレーニング
model_path = 'joint_reliability_model.pkl'
model = train_model(X, y, model_path)

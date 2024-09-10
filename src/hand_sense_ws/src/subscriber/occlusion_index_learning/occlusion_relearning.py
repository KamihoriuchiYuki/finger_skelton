import os
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

# モデルのトレーニング
def train_model(X, y, model_path):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

# モデルのロード
def load_model(model_path):
    return joblib.load(model_path)

# 既存モデルを用いた予測
def predict_with_existing_model(existing_model, X):
    # 既存モデルが学習時に使用していた特徴量だけに絞る
    columns_to_drop = ['existing_model_MCP', 'existing_model_PIP', 'existing_model_DIP', 'existing_model_TIP']
    X_for_prediction = X.drop(columns=columns_to_drop, errors='ignore')
    
    # 予測を実行
    return existing_model.predict(X_for_prediction)

# 既存モデルのバックアップ
def backup_existing_model(existing_model_path, backup_path):
    if os.path.exists(existing_model_path):
        os.rename(existing_model_path, backup_path)

# 学習データの準備
file_path = 'learning_data/combined_data_20240909_172908.csv'
data = load_data(file_path)
X, y = prepare_data(data)

# 既存モデルのファイルパスとバックアップファイルパス
existing_model_path = 'joint_reliability_model.pkl'
backup_model_path = 'joint_reliability_model_backup.pkl'

# 既存モデルのバックアップ
backup_existing_model(existing_model_path, backup_model_path)

# 既存モデルのロード
existing_model = load_model(backup_model_path)

# 既存モデルの予測結果を特徴量に追加
existing_model_predictions = predict_with_existing_model(existing_model, X)

# 予測結果の各列をXに追加
X['existing_model_MCP'] = existing_model_predictions[:, 0]
X['existing_model_PIP'] = existing_model_predictions[:, 1]
X['existing_model_DIP'] = existing_model_predictions[:, 2]
X['existing_model_TIP'] = existing_model_predictions[:, 3]

# 新たなモデルのトレーニング（ファイル名は既存モデルと同じ）
model = train_model(X, y, existing_model_path)

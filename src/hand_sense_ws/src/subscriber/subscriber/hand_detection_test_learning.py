import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle

# データをバッチ処理するためのジェネレータを定義
def batch_generator(data, batch_size=100):
    data_size = len(data)
    num_batches = data_size // batch_size
    for i in range(num_batches):
        yield data[i*batch_size:(i+1)*batch_size]
    if data_size % batch_size != 0:
        yield data[num_batches*batch_size:]

# データのロード
with open('depth_data.pkl', 'rb') as f:
    depth_data = pickle.load(f)
with open('keypoints_data.pkl', 'rb') as f:
    keypoints_data = pickle.load(f)

# データをシャッフル
depth_data, keypoints_data = shuffle(depth_data, keypoints_data, random_state=42)

# モデルの設定
model = RandomForestRegressor(n_estimators=100, random_state=42, warm_start=True)

# バッチ処理によるモデルの部分訓練
batch_size = 100  # メモリに応じてバッチサイズを調整
for batch_num, (X_batch, y_batch) in enumerate(zip(batch_generator(depth_data, batch_size), batch_generator(keypoints_data, batch_size))):
    X_batch = np.array(X_batch).reshape(len(X_batch), -1)  # バッチごとに整形
    y_batch = np.array(y_batch).reshape(len(y_batch), -1)  # バッチごとに整形
    if batch_num == 0:
        model.fit(X_batch, y_batch)
    else:
        model.n_estimators += 10  # 追加ツリーを増やす
        model.fit(X_batch, y_batch)

# モデルの保存
with open('hand_pose_model.pkl', 'wb') as f:
    pickle.dump(model, f)

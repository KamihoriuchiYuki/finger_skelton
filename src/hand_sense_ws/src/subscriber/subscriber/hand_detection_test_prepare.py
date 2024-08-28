import pickle
import numpy as np

# データのロード
with open('depth_data.pkl', 'rb') as f:
    depth_data = pickle.load(f)
with open('keypoints_data.pkl', 'rb') as f:
    keypoints_data = pickle.load(f)

# NaNを0に置換
keypoints_data = np.nan_to_num(keypoints_data, nan=0.0)

# クリーンなデータを保存
with open('clean_depth_data.pkl', 'wb') as f:
    pickle.dump(depth_data, f)
with open('clean_keypoints_data.pkl', 'wb') as f:
    pickle.dump(keypoints_data, f)

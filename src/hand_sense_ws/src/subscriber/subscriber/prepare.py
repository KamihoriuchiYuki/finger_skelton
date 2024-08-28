import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_data(data_dir):
    images = []
    depths = []
    labels = []

    for label, subdir in enumerate(['no_hand', 'hand']):
        subdir_path = os.path.join(data_dir, subdir)
        for filename in os.listdir(subdir_path):
            if filename.endswith('.png'):
                img_path = os.path.join(subdir_path, filename)
                depth_path = os.path.join(subdir_path, filename.replace('.png', '.npy'))

                image = cv2.imread(img_path)
                depth = np.load(depth_path)

                images.append(image)
                depths.append(depth)
                labels.append(label)

    images = np.array(images)
    depths = np.array(depths)
    labels = to_categorical(labels, num_classes=2)

    return images, depths, labels

data_dir = 'data'
X_rgb, X_depth, y = load_data(data_dir)

# データの正規化
X_rgb = X_rgb / 255.0
X_depth = X_depth / np.max(X_depth)

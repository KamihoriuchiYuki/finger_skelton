import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical, Sequence
import tensorflow as tf
from tensorflow.keras import layers, models

class DataGenerator(Sequence):
    def __init__(self, image_paths, depth_paths, labels, batch_size=8, dim=(240, 320), n_channels=3, n_classes=2, shuffle=True):
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        image_paths_temp = [self.image_paths[k] for k in indexes]
        depth_paths_temp = [self.depth_paths[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]
        X, y = self.__data_generation(image_paths_temp, depth_paths_temp, labels_temp)
        return X, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, image_paths_temp, depth_paths_temp, labels_temp):
        X_rgb = np.empty((self.batch_size, *self.dim, self.n_channels))
        X_depth = np.empty((self.batch_size, *self.dim, 1))
        y = np.empty((self.batch_size), dtype=int)

        for i, (img_path, depth_path) in enumerate(zip(image_paths_temp, depth_paths_temp)):
            image = cv2.imread(img_path)
            image = cv2.resize(image, (self.dim[1], self.dim[0]))  # 画像のサイズを変更
            depth = np.load(depth_path)
            depth = cv2.resize(depth, (self.dim[1], self.dim[0]))  # 深度マップのサイズを変更

            X_rgb[i,] = image / 255.0
            X_depth[i,] = np.expand_dims(depth, axis=-1) / np.max(depth)
            y[i] = labels_temp[i]
        
        return (X_rgb, X_depth), to_categorical(y, num_classes=self.n_classes)

def load_data_paths(data_dir):
    image_paths = []
    depth_paths = []
    labels = []

    for label, subdir in enumerate(['no_hand', 'hand']):
        subdir_path = os.path.join(data_dir, subdir)
        for filename in os.listdir(subdir_path):
            if filename.endswith('.png'):
                img_path = os.path.join(subdir_path, filename)
                depth_file = filename.replace('.png', '.npy').replace('rgb', 'depth')
                depth_path = os.path.join(subdir_path, depth_file)

                if os.path.exists(depth_path):
                    image_paths.append(img_path)
                    depth_paths.append(depth_path)
                    labels.append(label)

    return image_paths, depth_paths, labels

data_dir = 'input_data'
image_paths, depth_paths, labels = load_data_paths(data_dir)
batch_size = 8  # バッチサイズを小さくする

train_generator = DataGenerator(image_paths, depth_paths, labels, batch_size=batch_size, dim=(240, 320))

def create_model():
    input_rgb = layers.Input(shape=(240, 320, 3), name='rgb_input')
    input_depth = layers.Input(shape=(240, 320, 1), name='depth_input')

    # RGB画像のためのCNN
    x_rgb = layers.Conv2D(32, (3, 3), activation='relu')(input_rgb)
    x_rgb = layers.MaxPooling2D((2, 2))(x_rgb)
    x_rgb = layers.Conv2D(64, (3, 3), activation='relu')(x_rgb)
    x_rgb = layers.MaxPooling2D((2, 2))(x_rgb)
    x_rgb = layers.Conv2D(128, (3, 3), activation='relu')(x_rgb)
    x_rgb = layers.Flatten()(x_rgb)

    # 深度データのためのCNN
    x_depth = layers.Conv2D(32, (3, 3), activation='relu')(input_depth)
    x_depth = layers.MaxPooling2D((2, 2))(x_depth)
    x_depth = layers.Conv2D(64, (3, 3), activation='relu')(x_depth)
    x_depth = layers.MaxPooling2D((2, 2))(x_depth)
    x_depth = layers.Conv2D(128, (3, 3), activation='relu')(x_depth)
    x_depth = layers.Flatten()(x_depth)

    # 結合
    x = layers.Concatenate()([x_rgb, x_depth])
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(2, activation='softmax')(x)

    model = models.Model(inputs=[input_rgb, input_depth], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

# モデルのトレーニング
model.fit(train_generator, epochs=10)

# モデルを保存
model.save('hand_detection_model.h5')


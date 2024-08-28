import tensorflow as tf
import numpy as np
import os
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from realsense2_camera_msgs.msg import RGBD

height, width = 240, 320

class RsSub(Node):
    def __init__(self, model):
        super().__init__('hand_detector')
        self.bridge = CvBridge()
        self.model = model
        self.subscription = self.create_subscription(RGBD, '/camera/camera/rgbd', self.listener_callback, 10)

    def listener_callback(self, msg):
        array_rgb = self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8")
        array_depth = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")

        # 前処理
        input_rgb = cv2.resize(array_rgb, (width, height)) / 255.0
        input_depth = cv2.resize(array_depth, (width, height)) / np.max(array_depth)
        input_depth = np.expand_dims(input_depth, axis=-1)

        # モデル推論
        input_rgb = np.expand_dims(input_rgb, axis=0)
        input_depth = np.expand_dims(input_depth, axis=0)
        prediction = self.model.predict([input_rgb, input_depth])
        if np.argmax(prediction[0]) == 1:
            print("Hand detected")
        else:
            print("No hand detected")

# 学習済みモデルの読み込み
model = tf.keras.models.load_model('hand_detection_model.h5')

def main(args=None):
    rclpy.init(args=args)
    hand_detector = RsSub(model)
    rclpy.spin(hand_detector)
    hand_detector.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

import pickle
import numpy as np
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from realsense2_camera_msgs.msg import RGBD

# Load the trained model
with open('hand_pose_model.pkl', 'rb') as f:
    model = pickle.load(f)

class RsSub(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(RGBD, '/camera/camera/rgbd', self.listener_callback, 10)

    def listener_callback(self, msg):
        array_rgb = self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8")
        array_depth = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")
        predicted_keypoints = predict_keypoints(array_depth)
        for (x, y, z) in predicted_keypoints:
            cv2.circle(array_rgb, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.imshow("Image window2", array_rgb.astype(np.uint8))
        cv2.waitKey(1)
    
def predict_keypoints(depth_image):
    prediction = model.predict(depth_image.reshape(1, -1))
    return prediction.reshape(21, 3)

def main(args = None):
    rclpy.init(args = args)
    intel_subscriber = RsSub()
    rclpy.spin(intel_subscriber)
    intel_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
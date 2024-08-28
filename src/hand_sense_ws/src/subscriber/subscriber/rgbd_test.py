import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from realsense2_camera_msgs.msg import RGBD

min_depth, max_depth = 200, 500 # mm

class RsSub(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(RGBD, '/camera/camera/rgbd', self.listener_callback, 10)

    def listener_callback(self, msg):
        cv2.imshow("Image window", self.mask_rgb(self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8"), self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")))
        cv2.waitKey(1)

    def mask_rgb(self, rgb, depth) -> np.ndarray:
        mask = (depth <= min_depth) | (depth >= max_depth)
        return np.where(np.broadcast_to(mask[:, :, None], rgb.shape), 0, rgb).astype(np.uint8)

rclpy.init(args=None)
rclpy.spin(RsSub())
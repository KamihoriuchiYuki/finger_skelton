import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from realsense2_camera_msgs.msg import RGBD
import csv
import pprint

min_depth, max_depth = 100, 500 # mm
# hand_rgb = [150, 190, 200]

class RsSub(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(RGBD, '/camera/camera/rgbd', self.listener_callback, 10)

    def listener_callback(self, msg):
        array_rgb = self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8")
        array_depth = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")
        array_depth = self.depth_color(array_depth)
        ZERO = np.zeros((array_depth.shape[0], array_depth.shape[1], 3), dtype=np.uint8)
        ZERO[:, :, 0] = array_depth
        print(type(array_rgb))
        #print(array_rgb.shape)
        #print(ZERO.shape)
        #print(ZERO)
        cv2.imshow("Image window", ZERO)
        
        # print("C")
        cv2.waitKey(1)

    def depth_color(self, array_depth):
        array_depth = array_depth / 5
        array_depth = np.asarray(array_depth, dtype=int)
        array_depth = np.where((array_depth > 255), 0, 255-array_depth)
        #print(array_depth)
        return array_depth

        

def main(args = None):
    rclpy.init(args = args)
    intel_subscriber = RsSub()
    rclpy.spin(intel_subscriber)
    intel_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

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
        #array_red = array_rgb[:, :, 2]
        #array_green = array_rgb[:, :, 1]
        #array_blue = array_rgb[:, :, 0]
        cv2.imshow("Image window2", array_rgb.astype(np.uint8))
        array_mask_rgb = self.mask_rgb(array_rgb, array_depth)
        # array_mask_rgb = self.sentan(array_mask_rgb)
        #array_mask_rgb = self.choose_point(array_mask_rgb)
        #array_mask_red = array_mask_rgb[:, :, 2]
        #array_mask_green = array_mask_rgb[:, :, 1]
        #array_mask_blue = array_mask_rgb[:, :, 0]
        #array_cut_rgb_0 = self.cut_rgb(array_mask_red, array_mask_rgb)
        #array_cut_rgb_1 = self.cut_rgb(array_mask_green, array_cut_rgb_0)
        #array_cut_rgb_2 = self.cut_rgb(array_mask_blue, array_cut_rgb_1)
        #cv2.imshow("Image window", array_cut_rgb_2)
        #cv2.imshow("Image window2", array_mask_rgb)
        # cv2.imshow("Image window", self.choose_hand(array_red, array_green, array_blue, array_rgb))
        # array_xyz = self.trans_world(array_depth)
        # print(array_xyz)
        # print(array_mask_rgb)
        # array_cut_rgb = self.cut_rgb(array_red, array_green, array_blue, array_rgb)
        # array_cut_rgb_0 = array_cut_rgb[(array_mask_rgb != 0)]
        # array_cut_rgb_1 = array_cut_rgb_0.reshape(int(array_cut_rgb_0.size / 3), 3)
        # print(array_cut_rgb_1)
        # with open('rgb.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(array_cut_rgb_0)
        # print("C")
        cv2.waitKey(1)

    def mask_rgb(self, rgb, depth) -> np.ndarray:
        mask = (depth <= min_depth) | (depth >= max_depth)
        print(mask[:, :, None])
        #print(np.broadcast_to(mask[:, :, None], rgb.shape))
        return np.where(np.broadcast_to(mask[:, :, None], rgb.shape), 255, rgb).astype(np.uint8)
        
    def cut_rgb(self, color, rgb) -> np.ndarray:
        cut = (color == 255) 
        # print(np.broadcast_to(cut[:, :, None], rgb.shape))
        return np.where(np.broadcast_to(cut[:, :, None], rgb.shape), 255, rgb).astype(np.uint8)
    
    def choose_hand(self, r, g, b, rgb) -> np.ndarray:
        mask = (g <= 50) | (b <= r * 60 / 200) |(g >= r * 270 / 200) | (b >= r * 250 / 200) 
        return np.where(np.broadcast_to(mask[:, :, None], rgb.shape), 0, rgb).astype(np.uint8)
        
    def choose_point(self, rgb) -> np.ndarray:
        mask = (rgb >= 70) 
        return np.where(mask, 255, 0).astype(np.uint8)
        
        
    def sentan(self, rgb):
        num = 0
        shape = rgb.shape
        for i_0 in range(shape[0]):
           for i_1 in range(shape[1]):
               if num == 0:
                   if rgb[i_0, i_1, 0] != 0:
                       rgb[i_0, i_1, :] = [0, 0, 255]
                       num = 1
        return rgb  
        

def main(args = None):
    rclpy.init(args = args)
    intel_subscriber = RsSub()
    rclpy.spin(intel_subscriber)
    intel_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

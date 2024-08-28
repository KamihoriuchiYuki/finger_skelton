import cv2
from cv_bridge import CvBridge
import numpy as np
#import rospy
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from realsense2_camera_msgs.msg import RGBD
import mediapipe as mp
import csv
#import pprint
import time
import matplotlib.pyplot as plt
#from matplotlib import animation
import fcntl
import termios
import sys
import os
from datetime import datetime

n_1 = 0
n_2 = 0

class RsSub(Node):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    StartFlg = False
    rsp = ""
    FinishFlg = False
    hands = mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
    pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(RGBD, '/camera/camera/rgbd', self.listener_callback, 10)

    def listener_callback(self, msg):
        num_node = 21
        num_edge = 20
        array_rgb = self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8")
        array_depth = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")
        #image, finger = self.hand_skelton(array_rgb, num_node)
        image, body = self.body_skelton(array_rgb)
        #depth_order = self.order(finger[2, :], num_node)
        #image_shape = image.shape
        #print(finger[2, 8], finger[2, 12])
        #finger_pixel = self.to_pixel(finger, image_shape, num_node)
        cameraInfo = msg.depth_camera_info
        intrinsics = rs.intrinsics()
        intrinsics.width = cameraInfo.width
        intrinsics.height = cameraInfo.height
        intrinsics.ppx = cameraInfo.k[2]
        intrinsics.ppy = cameraInfo.k[5]
        intrinsics.fx = cameraInfo.k[0]
        intrinsics.fy = cameraInfo.k[4]
        #intrinsics.model = cameraInfo.distortion_model
        intrinsics.model  = rs.distortion.none     
        intrinsics.coeffs = [i for i in cameraInfo.d]
        #node_point = np.zeros((num_node, 3))
        #for i in range(num_node):
        #    node_point[i, :] = rs.rs2_deproject_pixel_to_point(intrinsics, finger_pixel[i, :], array_depth[finger_pixel[i, 1], finger_pixel[i, 0]])
        #print(node_point[8, :])
        #print(finger[2, :])
        
        cv2.imshow("Image window", image)
        #ims.append(image)
        #print(finger[:, 8])
        
        #print(nodes)
        #print(finger_1)
        cv2.waitKey(1) 
    
    def hand_skelton(self, image, num_node):
        finger = np.zeros((3, num_node))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image.shape)
        results = self.hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                for i in range(num_node):
                    finger[0, i] = hand_landmarks.landmark[i].x - 0.5
                    finger[1, i] = hand_landmarks.landmark[i].y - 0.5
                    finger[2, i] = hand_landmarks.landmark[i].z
                
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, finger
    
    def body_skelton(self, image):
        body = np.zeros((3, 1))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image.shape)
        results = self.pose.process(image)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing_styles.get_default_pose_landmarks_style())
            body[0, 0] = results.pose_landmarks.landmark[15].x
            body[1, 0] = results.pose_landmarks.landmark[15].y
            body[2, 0] = results.pose_landmarks.landmark[15].z
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, body

    def to_pixel(self, finger, shape, num_node):
        image_width, image_height = shape[1], shape[0]
        finger_pixel = np.zeros((num_node, 2), dtype=np.int64)
        for i in range(num_node):
            finger_pixel[i, 0] = int(finger[0, i] * image_width)
            finger_pixel[i, 1] = int(finger[0, i] * image_height)
        finger_pixel[:, 0] = self.cut_pixel(finger_pixel[:, 0], image_width)
        finger_pixel[:, 1] = self.cut_pixel(finger_pixel[:, 1], image_height)
        return finger_pixel

    def cut_pixel(self, pixel, max_pixel):
        pixel = np.where(pixel < 0, 0, pixel)
        pixel = np.where(pixel >= max_pixel, max_pixel - 1, pixel)
        return pixel
    
    def order(self, finger, num_node):
        depth_order = np.zeros((1, num_node), dtype=int)
        for i in range(num_node):
            if i != 0:
                depth_order[0, i] = i
                che = 0
            for j in range(i):
                if che == 0:
                    if finger[depth_order[0, j]] > finger[i]:
                        che = 1
                        depth_order[0, j], depth_order[0, i] = depth_order[0, i], depth_order[0, j]
                else:
                    depth_order[0, j], depth_order[0, i] = depth_order[0, i], depth_order[0, j]

        return depth_order
    
def main(args = None):
    rclpy.init(args = args)
    intel_subscriber = RsSub()
    #rclpy.spin(intel_subscriber)
    while True:
        rclpy.spin_once(intel_subscriber)
        key = getkey()
        #print(key)
        # enterで終了
        if key == 10:
            break
    #anim = animation.ArtistAnimation(cv2, ims)
    #anim.save("angle_finger.mp4", writer="ffmreg")
    
    intel_subscriber.destroy_node()
    rclpy.shutdown()

def getkey():
    fno = sys.stdin.fileno()

    #stdinの端末属性を取得
    attr_old = termios.tcgetattr(fno)

    # stdinのエコー無効、カノニカルモード無効
    attr = termios.tcgetattr(fno)
    attr[3] = attr[3] & ~termios.ECHO & ~termios.ICANON # & ~termios.ISIG
    termios.tcsetattr(fno, termios.TCSADRAIN, attr)

    # stdinをNONBLOCKに設定
    fcntl_old = fcntl.fcntl(fno, fcntl.F_GETFL)
    fcntl.fcntl(fno, fcntl.F_SETFL, fcntl_old | os.O_NONBLOCK)

    chr = 0

    try:
        # キーを取得
        c = sys.stdin.read(1)
        if len(c):
            while len(c):
                chr = (chr << 8) + ord(c)
                c = sys.stdin.read(1)
                #up:1792833
                #down:1792834
                #right:1792835
                #left:1792836
    finally:
        # stdinを元に戻す
        fcntl.fcntl(fno, fcntl.F_SETFL, fcntl_old)
        termios.tcsetattr(fno, termios.TCSANOW, attr_old)

    return chr


if __name__ == "__main__":
    main()
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
    StartFlg = False
    rsp = ""
    FinishFlg = False
    hands = mp_hands.Hands(
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
        image, finger = self.hand_skelton(array_rgb, num_node)
        im_size = 500
        cv2.imshow("Image window", image)
        #ims.append(image)
        #print(finger[:, 8])
        key_0 = getkey()
        global n_1
        global n_2
        #print(key_0)
        if key_0 == 1792833:
            if n_1 != 9:
                n_1 += 1
        if key_0 == 1792834:
            if n_1 != -9:
                n_1 -= 1
        if key_0 == 1792835:
            n_2 += 1
            if n_2 == 19:
                n_2 = -17
        if key_0 == 1792836:
            n_2 -= 1
            if n_2 == -18:
                n_2 = 18
        Sita_1 = n_1 * np.pi / 18
        Sita_2 = n_2 * np.pi / 18
        #print(n_2)
        M_1 = np.array([[np.cos(Sita_2), 0, -np.sin(Sita_2)], [0, 1, 0], [np.sin(Sita_2), 0, np.cos(Sita_2)]])
        M_2 = np.array([[1, 0, 0], [0, np.cos(Sita_1), -np.sin(Sita_1)], [0, np.sin(Sita_1), np.cos(Sita_1)]])
        rot_finger = M_2 @ M_1 @ finger
        finger_pixel = self.to_pixel(rot_finger, im_size, num_node)
        third_draw = np.zeros((im_size, im_size, 3), dtype=np.uint8)
        for i in range(num_node):
            cv2.circle(third_draw, (finger_pixel[i, 0], finger_pixel[i, 1]), 5, (0, 255, 0), -1)
        for i in range(num_edge):
            r = i % 4
            if r == 0:
                j_0 = 0
            else:
                j_0 = i
            j_1 = i + 1
            cv2.line(third_draw, (finger_pixel[j_0, 0], finger_pixel[j_0, 1]), (finger_pixel[j_1, 0], finger_pixel[j_1, 1]), (0, 255, 0), 3)
        cv2.imshow('3D model', third_draw)
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
    
    def to_pixel(self, finger, im_size, num_node):
        max_size = 2.0
        finger_pixel = np.zeros((num_node, 2), dtype=int)
        for i in range(num_node):
            x = finger[0, i] + max_size / 2
            y = finger[1, i] + max_size / 2
            finger_pixel[i, 0] = int(x / max_size * im_size)
            finger_pixel[i, 1] = int(y / max_size * im_size)
        return finger_pixel

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
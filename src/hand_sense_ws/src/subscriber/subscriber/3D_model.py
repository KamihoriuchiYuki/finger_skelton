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
#from matplotlib import pyplot as plt

n_1 = 0
n_2 = 0

min_depth, max_depth = 100, 500 # mm

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
        array_rgb = self.mask_rgb(array_rgb, array_depth)
        image, finger = self.hand_skelton(array_rgb, num_node)
        depth_order = self.order(finger[2, :], num_node)
        image_shape = image.shape
        #print(finger[2, 8], finger[2, 12])
        print(depth_order)
        finger_pixel = self.to_pixel(finger, image_shape, num_node)
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
        node_point = np.zeros((num_node, 3))
        for i in range(num_node):
            node_point[i, :] = rs.rs2_deproject_pixel_to_point(intrinsics, finger_pixel[i, :], array_depth[finger_pixel[i, 1], finger_pixel[i, 0]])
        #print(node_point[8, :])
        #print(finger[2, :])
        shield_list = self.shield(depth_order, node_point, num_node)
        cv2.imshow("Image window", image)
        #plt.imshow("Image window", image)
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
        M = np.dot(M_2, M_1)
        finger_1 = np.dot(M, finger)
        third_draw = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        nodes = np.zeros((num_node, 2), dtype =int)
        #cv2.circle(third_draw, (320, 240), 5, (255, 0, 0), -1)
        for i in range(num_node):
            x = finger_1[0, i] + 0.5
            y = finger_1[1, i] + 0.5
            #z = 10 + finger_1[2, i]
            #Sita_3 = np.arctan(y / z)
            #diag = np.sqrt(y ** 2 + z ** 2)
            #Sita_4 = np.arctan(x / diag)
            nodes[i, 0] = int(x * image_shape[1])
            nodes[i, 1] = int(y * image_shape[0])
            if x != 0.0:
                if y != 0.0:
                    if shield_list[0, i] == 0:
                        cv2.circle(third_draw, (nodes[i, 0], nodes[i, 1]), 5, (0, 255, 0), -1)
                    else:
                        cv2.circle(third_draw, (nodes[i, 0], nodes[i, 1]), 5, (0, 0, 255), -1)
        for i in range(num_edge):
            rmr = i % 4
            if rmr == 0:
                j_0 = 0
            else:
                j_0 = i
            j_1 = i + 1
            if finger_1[0, j_0] != 0.0:
                if finger_1[0, j_1] != 0.0:
                    cv2.line(third_draw, (nodes[j_0, 0], nodes[j_0, 1]), (nodes[j_1, 0], nodes[j_1, 1]), (0, 255, 0), 3)
        cv2.imshow('3D model', third_draw)
        #plt.imshow('3D model', third_draw)
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
    
    def shield(self, depth_order, node_point, num_node):
        print(depth_order)
        print(node_point)
        shield_list = np.zeros((1, num_node), dtype=int)
        alr_list = np.zeros((1, num_node), dtype=int)
        rad = 10.0
        for i in range(num_node):
            che = 0
            for j in range(i):
                if che == 0:
                    x_1 = node_point[depth_order[0, i], 0]
                    x_2 = node_point[depth_order[0, j], 0]
                    y_1 = node_point[depth_order[0, i], 1]
                    y_2 = node_point[depth_order[0, j], 1]
                    rx_1 = x_2 - x_1
                    ry_1 = y_2 - y_1
                    r = np.sqrt(rx_1 ** 2 + ry_1 ** 2)
                    if r < rad:
                        che = 1
                        shield_list[0, depth_order[0, i]] = 1
                    else:
                        if depth_order[0, j] == 0:
                            for k in range(5):
                                k_0 = k * 4 + 1
                                if alr_list[0, k_0] == 1:
                                    x_3 = node_point[depth_order[0, k_0], 0]
                                    y_3 = node_point[depth_order[0, k_0], 1]
                                    rx_2 = x_3 - x_1
                                    ry_2 = y_3 - y_1
                                    S = abs(rx_1 * ry_2 - rx_2 * ry_1)
                                    rx_3 = x_3 - x_2
                                    ry_3 = y_3 - y_2
                                    r_0 = np.sqrt(rx_3 ** 2 + ry_3 ** 2)
                                    if r_0 == 0.0:
                                        continue
                                    h = S / r_0
                                    if h < rad:
                                        r_1 = np.sqrt(rx_2 ** 2 + ry_2 ** 2)
                                        COS_1 = (r ** 2 + r_0 ** 2 - r_1 ** 2) / (2 * r * r_0)
                                        if COS_1 > 0.0:
                                            COS_2 = (r_1 ** 2 + r_0 ** 2 - r ** 2) / (2 * r_1 * r_0)
                                            if COS_2 > 0.0:
                                                che = 1
                                                shield_list[0, depth_order[0, i]] = 1
                                                break
                        else:
                            rmr_0 = depth_order[0, j] % 4
                            if rmr_0 == 0:
                                k_0 = depth_order[0, j] - 1
                                if alr_list[0, k_0] == 1:
                                    x_3 = node_point[depth_order[0, k_0], 0]
                                    y_3 = node_point[depth_order[0, k_0], 1]
                                    rx_2 = x_3 - x_1
                                    ry_2 = y_3 - y_1
                                    S = abs(rx_1 * ry_2 - rx_2 * ry_1)
                                    rx_3 = x_3 - x_2
                                    ry_3 = y_3 - y_2
                                    r_0 = np.sqrt(rx_3 ** 2 + ry_3 ** 2)
                                    if r_0 != 0.0:
                                        h = S / r_0
                                        if h < rad:
                                            r_1 = np.sqrt(rx_2 ** 2 + ry_2 ** 2)
                                            COS_1 = (r ** 2 + r_0 ** 2 - r_1 ** 2) / (2 * r * r_0)
                                            if COS_1 > 0.0:
                                                COS_2 = (r_1 ** 2 + r_0 ** 2 - r ** 2) / (2 * r_1 * r_0)
                                                if COS_2 > 0.0:
                                                    che = 1
                                                    shield_list[0, depth_order[0, i]] = 1
                                                    break
                            else:
                                for k in range(2):
                                    k_0 = depth_order[0, j] + 2 * k - 1
                                    if alr_list[0, k_0] == 1:
                                        x_3 = node_point[depth_order[0, k_0], 0]
                                        y_3 = node_point[depth_order[0, k_0], 1]
                                        rx_2 = x_3 - x_1
                                        ry_2 = y_3 - y_1
                                        S = abs(rx_1 * ry_2 - rx_2 * ry_1)
                                        rx_3 = x_3 - x_2
                                        ry_3 = y_3 - y_2
                                        r_0 = np.sqrt(rx_3 ** 2 + ry_3 ** 2)
                                        if r_0 == 0.0:
                                            continue
                                        h = S / r_0
                                        if h < rad:
                                            r_1 = np.sqrt(rx_2 ** 2 + ry_2 ** 2)
                                            COS_1 = (r ** 2 + r_0 ** 2 - r_1 ** 2) / (2 * r * r_0)
                                            if COS_1 > 0.0:
                                                COS_2 = (r_1 ** 2 + r_0 ** 2 - r ** 2) / (2 * r_1 * r_0)
                                                if COS_2 > 0.0:
                                                    che = 1
                                                    shield_list[0, depth_order[0, i]] = 1
                                                    break
            alr_list[0, i] = 1
        return shield_list
    
    def mask_rgb(self, rgb, depth) -> np.ndarray:
        mask = (depth <= min_depth) | (depth >= max_depth)
        print(mask[:, :, None])
        #print(np.broadcast_to(mask[:, :, None], rgb.shape))
        return np.where(np.broadcast_to(mask[:, :, None], rgb.shape), 0, rgb).astype(np.uint8)


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
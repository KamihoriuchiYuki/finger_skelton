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
import pprint
import time
import matplotlib.pyplot as plt
from matplotlib import animation
import fcntl
import termios
import sys
import os

f = open("data.csv", "w", newline="")
writer = csv.writer(f)
title_list = []
finger_list = ['time_count', 'THUMB_', 'INDEX_', 'MIDDLE_', 'RING_', 'PINKY_']
title_list.append(finger_list[0])
count = 1
finger_count = 1
for i in range(15):
    string = finger_list[finger_count] + str(4 - count)
    title_list.append(string)
    if count == 3:
        count = 0
        finger_count += 1
    count += 1
writer.writerow(title_list)
start_time = time.perf_counter()

class RsSub(Node):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    StartFlg = False
    rsp = ""
    FinishFlg = False
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.bridge = CvBridge()
        with self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            self.subscription = self.create_subscription(RGBD, '/camera/camera/rgbd', self.listener_callback, 10)

    def listener_callback(self, msg):
        num_node = 21
        num_edge = 20
        num_joint = 15
        array_rgb = self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8")
        array_depth = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")
        image, index_finger = self.hand_skelton(array_rgb, num_node)
        #cv2.imshow("Image window", array_rgb)
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
            node_point[i, :] = rs.rs2_deproject_pixel_to_point(intrinsics, index_finger[i, :], array_depth[index_finger[i, 1], index_finger[i, 0]])
        #print(node_point)
        edge = np.zeros((num_edge, 3))
        for i in range(num_edge):
            r = i % 4
            if r == 0:
                j_0 = 0
            else:
                j_0 = i
            j_1 = i + 1
            edge_0 = []
            for a, b in zip(node_point[j_1, :], node_point[j_0, :]):
                edge_0.append(a - b)
            edge[i, :] = edge_0
        angle_joint = np.zeros(num_joint + 1)
        angle_joint[0] = time.perf_counter() - start_time
        count_joint = 0
        for i in range(num_joint):
            r = i % 3
            if r == 0:
                count_joint += 1
            j_0 = count_joint - 1
            j_1 = count_joint
            v_finger_0 = -edge[j_0, :]
            v_finger_1 = edge[j_1, :]
            v_dot = np.dot(v_finger_0, v_finger_1)
            v_abs_0 = np.sqrt(np.dot(v_finger_0, v_finger_0))
            v_abs_1 = np.sqrt(np.dot(v_finger_1, v_finger_1))
            v_abs = v_abs_0 * v_abs_1
            COS = v_dot / v_abs
            SITA = np.arccos(COS)
            angle_joint[i + 1] = 180 - SITA * 180 / np.pi
            count_joint += 1
        #print("angle_finger", angle_joint)
        writer.writerow(angle_joint)
        cv2.imshow("Image window", image)
        #ims.append(image)
        #test()
        cv2.waitKey(5) 
    
    def hand_skelton(self, image, num_node):
        image_width, image_height = image.shape[1], image.shape[0]
        finger = np.zeros((num_node, 2), dtype=np.int64)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image.shape)
        global hands
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                for i in range(num_node):
                    finger[i, 0] = int(hand_landmarks.landmark[i].x * image_width)
                    finger[i, 1] = int(hand_landmarks.landmark[i].y * image_height)
                finger[:, 0] = self.cut_pixel(finger[:, 0], image_width)
                finger[:, 1] = self.cut_pixel(finger[:, 1], image_height)
            
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, finger
    
    def cut_pixel(self, pixel, max_pixel):
        pixel = np.where(pixel < 0, 0, pixel)
        pixel = np.where(pixel >= max_pixel, max_pixel - 1, pixel)
        return pixel

def main(args = None):
    rclpy.init(args = args)
    intel_subscriber = RsSub()
    #rclpy.spin(intel_subscriber)
    while True:
        rclpy.spin_once(intel_subscriber)
        key = getkey()
        # enterで終了
        if key == 10:
            break
    #anim = animation.ArtistAnimation(cv2, ims)
    #anim.save("angle_finger.mp4", writer="ffmreg")
    f.close()
    fig = plt.figure(figsize=[20, 5])
    f_0 = open("data.csv")
    reader = csv.reader(f_0)
    data = [row for row in reader]
    data = np.array(data)
    data = data.T
    length = data.shape[1]
    color_list = ["black", "dimgray", "darkgray", "brown", "red", "lightsalmon", "goldenrod", "yellow", "orange", "darkgreen", "limegreen", "yellowgreen", "purple", "blue", "aqua"]
    for i in range(15):
        plt.plot([float(v) for v in data[0, 1:length]], [float(v_0) for v_0 in data[i+1, 1:length]], color = color_list[i], label = data[i+1, 0], linewidth = 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.xlabel("time[s]")
    plt.ylabel("angle[°]")
    plt.title("angle of finger")
    fig.savefig("angle_finger.png")
    f_0.close()
    plt.close()
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
    finally:
        # stdinを元に戻す
        fcntl.fcntl(fno, fcntl.F_SETFL, fcntl_old)
        termios.tcsetattr(fno, termios.TCSANOW, attr_old)

    return chr

def test():
    print(1)

if __name__ == "__main__":
    main()

import matplotlib
matplotlib.use('Agg')  # 非GUIバックエンドに切り替える
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from realsense2_camera_msgs.msg import RGBD
import mediapipe as mp
import csv
import time
import fcntl
import termios
import sys
import os

# ファイルオープンとCSVライター設定
f = open("data.csv", "w", newline="")
writer = csv.writer(f)
title_list = ['time_count', 'INDEX_3', 'INDEX_2', 'INDEX_1', 'INDEX_0']
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
        super().__init__('angle_finger')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(RGBD, '/camera/camera/rgbd', self.listener_callback, 10)
    
    def listener_callback(self, msg):
        num_node = 21  # 手の全ての関節の数
        array_rgb = self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8")
        array_depth = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")
        image, index_finger = self.hand_skelton(array_rgb, num_node)
        cameraInfo = msg.depth_camera_info
        intrinsics = rs.intrinsics()
        intrinsics.width = cameraInfo.width
        intrinsics.height = cameraInfo.height
        intrinsics.ppx = cameraInfo.k[2]
        intrinsics.ppy = cameraInfo.k[5]
        intrinsics.fx = cameraInfo.k[0]
        intrinsics.fy = cameraInfo.k[4]
        intrinsics.model = rs.distortion.none     
        intrinsics.coeffs = [i for i in cameraInfo.d]

        # 人差し指の関節の距離を保存
        index_distances = [time.perf_counter() - start_time]
        for i in range(5, 9):  # 人差し指のランドマークの範囲
            point_depth = array_depth[index_finger[i, 1], index_finger[i, 0]]
            distance = rs.rs2_deproject_pixel_to_point(intrinsics, index_finger[i, :], point_depth)
            distance_from_camera = np.linalg.norm(distance)
            index_distances.append(distance_from_camera)

        writer.writerow(index_distances)
        cv2.imshow("Image window", image)
        cv2.waitKey(5) 
    
    def hand_skelton(self, image, num_node):
        image_width, image_height = image.shape[1], image.shape[0]
        finger = np.zeros((num_node, 2), dtype=np.int64)
        with self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    while True:
        rclpy.spin_once(intel_subscriber)
        key = getkey()
        if key == 10:  # Enterキーで終了
            break
    
    f.close()
    
    # CSVからデータを読み込み、グラフを描画
    fig = plt.figure(figsize=[10, 5])
    f_0 = open("data.csv")
    reader = csv.reader(f_0)
    data = [row for row in reader]
    data = np.array(data)
    data = data.T
    length = data.shape[1]
    color_list = ["blue", "green", "orange", "red"]  # 人差し指の関節ごとの色
    for i in range(1, 5):
        plt.plot([float(v) for v in data[0, 1:length]], [float(v_0) for v_0 in data[i, 1:length]], color = color_list[i-1], label = title_list[i], linewidth = 1)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.xlabel("time[s]")
    plt.ylabel("distance from camera[m]")
    plt.title("Distance of Index Finger Joints from Camera")
    fig.savefig("index_finger_distance.png")
    f_0.close()
    plt.close()

    intel_subscriber.destroy_node()
    rclpy.shutdown()

def getkey():
    fno = sys.stdin.fileno()
    attr_old = termios.tcgetattr(fno)
    attr = termios.tcgetattr(fno)
    attr[3] = attr[3] & ~termios.ECHO & ~termios.ICANON
    termios.tcsetattr(fno, termios.TCSADRAIN, attr)
    fcntl_old = fcntl.fcntl(fno, fcntl.F_GETFL)
    fcntl.fcntl(fno, fcntl.F_SETFL, fcntl_old | os.O_NONBLOCK)

    chr = 0
    try:
        c = sys.stdin.read(1)
        if len(c):
            while len(c):
                chr = (chr << 8) + ord(c)
                c = sys.stdin.read(1)
    finally:
        fcntl.fcntl(fno, fcntl.F_SETFL, fcntl_old)
        termios.tcsetattr(fno, termios.TCSANOW, attr_old)
    return chr

if __name__ == "__main__":
    main()

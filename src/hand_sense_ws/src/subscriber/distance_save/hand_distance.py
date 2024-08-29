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
import matplotlib
matplotlib.use('Agg')  # これを追加して、非GUIバックエンドに切り替える
import matplotlib.pyplot as plt
from std_msgs.msg import Float64MultiArray

# ファイルオープンとタイトル行の書き込み
f = open("hand_distances_data.csv", "w", newline="")
writer = csv.writer(f)
title_list = ['time_count']
finger_joints = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
                 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
                 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
                 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
                 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

# 各関節のカメラからの距離を保存するためのタイトルを作成
for joint in finger_joints:
    title_list.append(f"{joint}_distance")
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
        super().__init__('hand_landmarks_recorder')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(RGBD, '/camera/camera/rgbd', self.listener_callback, 10)
        self.pub = self.create_publisher(Float64MultiArray, 'hand_landmarks_topic', 10)

    def listener_callback(self, msg):
        num_node = 21
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
        intrinsics.model  = rs.distortion.none     
        intrinsics.coeffs = [i for i in cameraInfo.d]

        node_point = np.zeros((num_node, 3))
        landmark_data = [time.perf_counter() - start_time]  # 時間データを格納

        for i in range(num_node):
            # 3D空間の各関節の座標を取得
            node_point[i, :] = rs.rs2_deproject_pixel_to_point(intrinsics, index_finger[i, :], array_depth[index_finger[i, 1], index_finger[i, 0]])
            # カメラからの距離を計算（x, y, z の 3次元ベクトルの大きさ）
            distance = np.linalg.norm(node_point[i, :])
            # 各関節の距離を記録
            landmark_data.append(distance)

        writer.writerow(landmark_data)
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

def plot_and_save_graph():
    # CSVファイルを開いてデータを読み込む
    with open("hand_distances_data.csv", "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    # タイトル行とデータ行を分離
    headers = data[0]
    data = np.array(data[1:], dtype=np.float32).T

    # 時間データ
    time_data = data[0]

    # グラフを作成
    plt.figure(figsize=(20, 10))
    for i in range(1, len(headers)):
        plt.plot(time_data, data[i], label=headers[i])

    plt.xlabel("Time [s]")
    plt.ylabel("Distance from Camera [m]")
    plt.ylim([100, 1000])
    plt.title("Distance of Finger Joints from Camera Over Time")
    plt.legend(loc="upper right")
    plt.grid(True)
    
    # グラフを画像ファイルとして保存
    plt.savefig("hand_distances_over_time.png")
    plt.close()

def main(args = None):
    rclpy.init(args = args)
    intel_subscriber = RsSub()
    while True:
        rclpy.spin_once(intel_subscriber)
        key = getkey()
        if key == 10:
            break

    f.close()
    plot_and_save_graph()  # グラフを作成して保存
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

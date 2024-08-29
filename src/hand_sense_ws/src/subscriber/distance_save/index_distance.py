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
import matplotlib.pyplot as plt
from matplotlib import animation
import fcntl
import termios
import sys
import os
from std_msgs.msg import Float64MultiArray

class RsSub(Node):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    StartFlg = False
    recording = False  # Toggle for recording data
    start_time = None
    f = None
    writer = None

    def __init__(self):
        super().__init__('angle_finger')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(RGBD, '/camera/camera/rgbd', self.listener_callback, 10)
        self.pub = self.create_publisher(Float64MultiArray, 'finger_angle_topic', 10)

    def listener_callback(self, msg):
        num_node = 21
        num_joint = 4  # Only index finger joints (MCP, PIP, DIP, TIP)
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
        distances = []
        for i in range(5, 9):  # Only index finger joints
            point = rs.rs2_deproject_pixel_to_point(intrinsics, index_finger[i, :], array_depth[index_finger[i, 1], index_finger[i, 0]])
            distance = np.linalg.norm(point)
            distances.append(distance)
        
        if self.recording:
            current_time = time.perf_counter() - self.start_time
            data_row = [current_time] + distances
            self.writer.writerow(data_row)

        msg_data = Float64MultiArray()
        msg_data.data = distances
        self.pub.publish(msg_data)

        cv2.imshow("Image window", image)
        cv2.waitKey(5)

    def hand_skelton(self, image, num_node):
        image_width, image_height = image.shape[1], image.shape[0]
        finger = np.zeros((num_node, 2), dtype=np.int64)
        with self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
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

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            self.start_time = time.perf_counter()
            self.f = open("index_finger_data.csv", "w", newline="")
            self.writer = csv.writer(self.f)
            header = ["time[s]", "MCP", "PIP", "DIP", "TIP"]
            self.writer.writerow(header)
        else:
            self.f.close()
            self.plot_data()

    def plot_data(self):
        fig = plt.figure(figsize=[20, 5])
        with open("index_finger_data.csv") as f_0:
            reader = csv.reader(f_0)
            data = [row for row in reader]
            data = np.array(data)
            data = data.T
            length = data.shape[1]
            plt.plot([float(v) for v in data[0, 1:length]], [float(v_0) for v_0 in data[1, 1:length]], color="red", label="MCP", linewidth=1)
            plt.plot([float(v) for v in data[0, 1:length]], [float(v_0) for v_0 in data[2, 1:length]], color="green", label="PIP", linewidth=1)
            plt.plot([float(v) for v in data[0, 1:length]], [float(v_0) for v_0 in data[3, 1:length]], color="blue", label="DIP", linewidth=1)
            plt.plot([float(v) for v in data[0, 1:length]], [float(v_0) for v_0 in data[4, 1:length]], color="purple", label="TIP", linewidth=1)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            plt.xlabel("time[s]")
            plt.ylabel("distance[m]")
            plt.title("Distance of Index Finger Joints from Camera")
            fig.savefig("index_finger_distance.png")
            plt.close()

def main(args = None):
    rclpy.init(args = args)
    intel_subscriber = RsSub()
    while True:
        rclpy.spin_once(intel_subscriber)
        key = getkey()
        if key == 10:
            break
        elif key == ord('s'):
            intel_subscriber.toggle_recording()
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

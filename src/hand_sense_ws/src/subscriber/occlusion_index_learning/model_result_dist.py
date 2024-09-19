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
matplotlib.use('Agg')  # Non-GUI backend for matplotlib
import matplotlib.pyplot as plt
from std_msgs.msg import Float64MultiArray
import joblib

class RsSub(Node):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    StartFlg = False
    recording = False  # Toggle for recording data
    start_time = None
    f = None
    writer = None
    pos_f = None
    pos_writer = None
    video_out = None
    bef_p = np.zeros(4, dtype=np.int16)
    saveFlg = False

    def __init__(self):
        super().__init__('angle_finger')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(RGBD, '/camera/camera/rgbd', self.listener_callback, 10)
        self.pub = self.create_publisher(Float64MultiArray, 'finger_angle_topic', 10)
        self.model = joblib.load('joint_reliability_model_dist.pkl')

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

        # Collect 2D coordinates and corresponding depth values
        pixel_depths = []
        pixel_positions = []
        joint_positions = []

        # Get wrist depth value
        wrist_x, wrist_y = index_finger[0, 0], index_finger[0, 1]
        if 0 <= wrist_x < array_depth.shape[1] and 0 <= wrist_y < array_depth.shape[0]:
            wrist_depth = array_depth[wrist_y, wrist_x]

        pixel_positions.append((wrist_x, wrist_y, wrist_depth))  # Add wrist (x, y, depth)

        wrist = rs.rs2_deproject_pixel_to_point(intrinsics, index_finger[0, :], array_depth[index_finger[0, 1], index_finger[0, 0]])

        # Get depth values for index finger joints (MCP, PIP, DIP, TIP)
        for i in range(5, 9):  # Only index finger joints
            x, y = index_finger[i, 0], index_finger[i, 1]
            
            # Depth value check and handling
            if 0 <= x < array_depth.shape[1] and 0 <= y < array_depth.shape[0]:
                depth_value = array_depth[y, x]
                if np.isfinite(depth_value):  # Check if the depth value is valid (finite number)
                    pixel_depths.append(float(depth_value))  # Ensure the value is of type float
                else:
                    depth_value = 0.0
                    pixel_depths.append(depth_value)
            else:
                depth_value = 0.0
                pixel_depths.append(depth_value)
            
            pixel_positions.append((x, y, depth_value))  # Save (x, y, depth)

            point = rs.rs2_deproject_pixel_to_point(intrinsics, index_finger[i, :], array_depth[index_finger[i, 1], index_finger[i, 0]])
            for j in range(3):
                point[j] -= wrist[j]
            joint_positions.append(point)
        
        if self.recording:
            current_time = time.perf_counter() - self.start_time
            data_row = [current_time] + pixel_depths
            self.writer.writerow(data_row)
            pos_now = [coord for pos in pixel_positions for coord in pos] + [coord for pos in joint_positions for coord in pos]
            if not self.saveFlg:
                self.bef_pos = pos_now
                self.saveFlg = not self.saveFlg
            diff_row = [x - y for x, y in zip(pos_now, self.bef_pos)]
            array_pos_now = np.array(pos_now).flatten().reshape(1, -1)
            array_diff_row = np.array(diff_row).flatten().reshape(1, -1)
            array_bef_p = np.array(self.bef_p).flatten().reshape(1, -1)
            features = np.concatenate([array_pos_now, array_diff_row, array_bef_p], axis=1)
            #print(features.shape)
            predictions = self.model.predict(features)

            # 信頼値に基づき色を決定
            colors = [(0, 255, 0) if p == 0 else (0, 0, 255) for p in predictions[0]]
            self.draw_joints(image, index_finger, colors)

            self.bef_p = predictions
            self.bef_pos = pos_now

            pos_row = [current_time] + pos_now
            self.pos_writer.writerow(pos_row)
            
            # 映像を保存
            self.video_out.write(image)

        msg_data = Float64MultiArray()
        msg_data.data = pixel_depths
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
    
    def draw_joints(self, image, index_finger, colors):
        for i, color in enumerate(colors):
            cv2.circle(image, tuple(index_finger[i+5]), 5, color, -1)

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.start_time = time.perf_counter()
            self.f = open(f"data_index_2/index_finger_data_{timestamp}.csv", "w", newline="")
            self.writer = csv.writer(self.f)
            header = ["time[s]", "MCP", "PIP", "DIP", "TIP"]
            self.writer.writerow(header)

            self.pos_f = open(f"data_index_2/finger_joint_positions_{timestamp}.csv", "w", newline="")
            self.pos_writer = csv.writer(self.pos_f)
            pos_header = ["time[s]", "Wrist_X", "Wrist_Y", "Wrist_D",
                          "MCP_X", "MCP_Y", "MCP_D",
                          "PIP_X", "PIP_Y", "PIP_D",
                          "DIP_X", "DIP_Y", "DIP_D",
                          "TIP_X", "TIP_Y", "TIP_D",
                          "MCP_RX", "MCP_RY", "MCP_RZ",
                          "PIP_RX", "PIP_RY", "PIP_RZ",
                          "DIP_RX", "DIP_RY", "DIP_RZ",
                          "TIP_RX", "TIP_RY", "TIP_RZ",
                          "MCP_Distance", "PIP_Distance", "DIP_Distance", "TIP_Distance"]
            self.pos_writer.writerow(pos_header)
            
            # 映像の保存設定
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_out = cv2.VideoWriter(f"data_index_2/index_finger_video_{timestamp}.avi", fourcc, 20.0, (640, 480))

        else:
            self.f.close()
            self.pos_f.close()
            self.video_out.release()  # 映像の保存を終了
            self.plot_data()

    def plot_data(self):
        fig, axs = plt.subplots(1, 1, figsize=[20, 5])
        with open(self.f.name) as f_0:
            reader = csv.reader(f_0)
            data = [row for row in reader]
            data = np.array(data)
            data = data.T
            length = data.shape[1]
            axs.plot([float(v) for v in data[0, 1:length]], [float(v_0) for v_0 in data[1, 1:length]], color="red", label="MCP", linewidth=1)
            axs.plot([float(v) for v in data[0, 1:length]], [float(v) for v in data[2, 1:length]], color="green", label="PIP", linewidth=1)
            axs.plot([float(v) for v in data[0, 1:length]], [float(v) for v in data[3, 1:length]], color="blue", label="DIP", linewidth=1)
            axs.plot([float(v) for v in data[0, 1:length]], [float(v) for v in data[4, 1:length]], color="purple", label="TIP", linewidth=1)
            axs.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            axs.set_xlabel("time[s]")
            axs.set_ylabel("distance[mm]")
            axs.set_ylim([200, 500])
            axs.set_title("Distance of Index Finger Joints from Camera")
            fig.savefig(f"data_index_2/index_finger_distance_{time.strftime('%Y%m%d_%H%M%S')}.png")
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
    except IOError:
        pass
    finally:
        termios.tcsetattr(fno, termios.TCSADRAIN, attr_old)
        fcntl.fcntl(fno, fcntl.F_SETFL, fcntl_old)
    return chr

if __name__ == '__main__':
    main()

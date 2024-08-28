import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from realsense2_camera_msgs.msg import RGBD
import mediapipe as mp
import csv
import pprint


class RsSub(Node):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(RGBD, '/camera/camera/rgbd', self.listener_callback, 10)

    def listener_callback(self, msg):
        array_rgb = self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8")
        array_depth = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")
        image, index_finger = self.hand_skelton(array_rgb)
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
        point_0 = rs.rs2_deproject_pixel_to_point(intrinsics, index_finger[0, :], array_depth[index_finger[0, 1], index_finger[0, 0]])
        point_1 = rs.rs2_deproject_pixel_to_point(intrinsics, index_finger[1, :], array_depth[index_finger[1, 1], index_finger[1, 0]])
        point_2 = rs.rs2_deproject_pixel_to_point(intrinsics, index_finger[2, :], array_depth[index_finger[2, 1], index_finger[2, 0]])
        v_finger_0 = []
        v_finger_1 = []
        for a, b in zip(point_0, point_1):
            v_finger_0.append(a - b)
        for a, b in zip(point_2, point_1):
            v_finger_1.append(a - b)
        v_dot = np.dot(v_finger_0, v_finger_1)
        v_abs_0 = np.sqrt(np.dot(v_finger_0, v_finger_0))
        v_abs_1 = np.sqrt(np.dot(v_finger_1, v_finger_1))
        COS = v_dot / (v_abs_0 * v_abs_1)
        SITA = np.arccos(COS)
        angle_finger = 180 - SITA * 180 / np.pi
        print(angle_finger)
        cv2.imshow("Image window", image)
        cv2.waitKey(1) 
    
    def hand_skelton(self, image):
        image_width, image_height = image.shape[1], image.shape[0]
        index_finger = np.zeros((3, 2), dtype=np.int64)
        with self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #print(image.shape)
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                    index_finger[0, 0] = int(hand_landmarks.landmark[7].x * image_width)
                    index_finger[0, 1] = int(hand_landmarks.landmark[7].y * image_height)
                    index_finger[1, 0] = int(hand_landmarks.landmark[6].x * image_width)
                    index_finger[1, 1] = int(hand_landmarks.landmark[6].y * image_height)
                    index_finger[2, 0] = int(hand_landmarks.landmark[5].x * image_width)
                    index_finger[2, 1] = int(hand_landmarks.landmark[5].y * image_height)
                    index_finger[:, 0] = self.cut_pixel(index_finger[:, 0], image_width)
                    index_finger[:, 1] = self.cut_pixel(index_finger[:, 1], image_height)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, index_finger
    
    def cut_pixel(self, pixel, max_pixel):
        pixel = np.where(pixel < 0, 0, pixel)
        pixel = np.where(pixel >= max_pixel, max_pixel - 1, pixel)
        return pixel

def main(args = None):
    rclpy.init(args = args)
    intel_subscriber = RsSub()
    rclpy.spin(intel_subscriber)
    intel_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

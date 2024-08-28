import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from realsense2_camera_msgs.msg import RGBD
import mediapipe as mp
import pickle

class RsSub(Node):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(RGBD, '/camera/camera/rgbd', self.listener_callback, 10)

        # データ収集用リスト
        self.depth_data = []
        self.keypoints_data = []

    def listener_callback(self, msg):
        array_rgb = self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8")
        array_depth = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")
        image, keypoints = self.hand_skelton(array_rgb, array_depth)
        
        # データ収集
        if array_depth.shape == (480, 640):
            self.depth_data.append(array_depth)
            self.keypoints_data.append(keypoints)

        cv2.imshow("Image window", image)
        cv2.waitKey(1) 
    
    def hand_skelton(self, image, depth_image):
        image_width, image_height = image.shape[1], image.shape[0]
        keypoints = []
        with self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                    
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_height)
                        z = depth_image[y, x] if 0 <= x < image_width and 0 <= y < image_height else np.nan
                        if np.isnan(x):
                            x = -1
                            y = -1
                            z = -1.0
                        if np.isnan(y):
                            x = -1
                            y = -1
                            z = -1.0
                        if np.isnan(z):
                            x = -1
                            y = -1
                            z = -1.0
                        keypoints.append([x, y, z])
            else:
                # 手が検出されなかった場合、NaNの座標を追加
                keypoints = [[-1, -1, -1.0] for _ in range(21)]
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(keypoints)
        
        return image, keypoints

    def save_data(self):
        # Ensure that depth_data and keypoints_data are consistent
        depth_data = np.array(self.depth_data)
        keypoints_data = np.array(self.keypoints_data, dtype=object)

        with open('depth_data.pkl', 'wb') as f:
            pickle.dump(depth_data, f)
        with open('keypoints_data.pkl', 'wb') as f:
            pickle.dump(keypoints_data, f)

def main(args=None):
    rclpy.init(args=args)
    intel_subscriber = RsSub()
    try:
        rclpy.spin(intel_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        intel_subscriber.save_data()
        intel_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()


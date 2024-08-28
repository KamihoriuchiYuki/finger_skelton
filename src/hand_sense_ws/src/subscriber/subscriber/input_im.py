import numpy as np
import os
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from realsense2_camera_msgs.msg import RGBD
import fcntl
import termios
import sys

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(RGBD, '/camera/camera/rgbd', self.listener_callback, 10)
        self.image_count = 0
        self.save_dir = 'input_data'
        self.hand_dir = os.path.join(self.save_dir, 'hand')
        self.no_hand_dir = os.path.join(self.save_dir, 'no_hand')
        os.makedirs(self.hand_dir, exist_ok=True)
        os.makedirs(self.no_hand_dir, exist_ok=True)

    def listener_callback(self, msg):
        array_rgb = self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8")
        array_depth = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")
        cv2.imshow("Image window2", array_rgb.astype(np.uint8))
        key = cv2.waitKey(1)
        if key == ord('y'):
            print('i')
            cv2.imwrite(os.path.join(self.hand_dir, f'rgb_{self.image_count}.png'), array_rgb)
            np.save(os.path.join(self.hand_dir, f'depth_{self.image_count}.npy'), array_depth)
        #else:
            #cv2.imwrite(os.path.join(self.no_hand_dir, f'rgb_{self.image_count}.png'), array_rgb)
            #np.save(os.path.join(self.no_hand_dir, f'depth_{self.image_count}.npy'), array_depth)
        self.image_count += 1

def main(args=None):
    rclpy.init(args=args)
    data_collector = DataCollector()
    rclpy.spin(data_collector)
    data_collector.destroy_node()
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

if __name__ == "__main__":
    main()

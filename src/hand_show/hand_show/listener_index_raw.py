import cv2
import numpy as np
#import rospy
import rclpy
from rclpy.node import Node
import fcntl
import termios
import sys
import os
from std_msgs.msg import Float64MultiArray #int型のメッセージのimport

# C++と同じく、Node型を継承します。
class HandShow(Node):
    # コンストラクタです、HandShowクラスのインスタンスを作成する際に呼び出されます。
    def __init__(self):
        # 継承元のクラスを初期化します。（https://www.python-izm.com/advanced/class_extend/）今回の場合継承するクラスはNodeになります。
        super().__init__('hand_show')
        # Subscriptionを作成、Publishしたデータを受け取ることが出来るようにします。
        self.subscription = self.create_subscription(Float64MultiArray, '/combined_raw', self.listener_callback, 10)
        self.num_node = 21
        self.num_fin = 5
        self.num_input = 15
        self.im_height = 1280
        self.im_width = 1280
        self.Sita_0 = np.pi / 2
        self.Sita_1 = np.pi / 2

        # 各指の長さを格納
        self.fin_length = np.array([0.5, 1.0, 0.8, 0.6, 1.5, 1.0, 0.8, 0.5, 1.5, 1.0, 0.8, 0.6, 1.5, 1.0, 0.8, 0.5, 1.5, 0.8, 0.6, 0.5])

        self.node_point = np.zeros((3, self.num_node))
        self.angle_0 = np.array([7 * np.pi / 36, np.pi / 2, 7 * np.pi / 12, np.pi * 2 / 3, np.pi * 3 / 4])

        for i in range(self.num_fin):
            self.node_point[0, 4*i+1] = self.fin_length[4*i] * np.cos(self.angle_0[i])
            self.node_point[1, 4*i+1] = self.fin_length[4*i] * np.sin(self.angle_0[i])

    # メッセージを受け取ったときに実行されるコールバック関数
    def listener_callback(self, msg_data):
        # loggerを使ってterminalに文字列を出力する。
        angle_index = msg_data.data
        angle_index = np.array(angle_index)
        angle = np.zeros(self.num_input)
        angle[0] = 7 * np.pi / 36
        angle[1] = np.pi / 2
        angle[2] = np.pi / 2
        # angle[3] = angle_index[3]*np.pi / 180
        # angle[4] = angle_index[4]*np.pi / 180
        # angle[5] = angle_index[5]*np.pi / 180
        angle[3] = angle_index[4]*np.pi / 180
        angle[4] = angle_index[5]*np.pi / 180
        angle[5] = angle_index[6]*np.pi / 180
        angle[6] = 5 * np.pi / 12
        angle[7] = np.pi / 2
        angle[8] = 5 * np.pi / 12
        angle[9] = 5 * np.pi / 12
        angle[10] = np.pi / 2
        angle[11] = 5 * np.pi / 12
        angle[12] = 5 * np.pi / 12
        angle[13] = np.pi / 2
        angle[14] = 5 * np.pi / 12
        sum_angle = 0.0
        fin_count = 1
        for i in range(self.num_input):
            if i == 0:
                fin_angle = self.angle_0[0] + angle[0]
                self.node_point[0, 2] = self.fin_length[1] * np.cos(fin_angle) + self.node_point[0, 1]
                self.node_point[1, 2] = self.fin_length[1] * np.sin(fin_angle) + self.node_point[1, 1]
            else:
                if i % 3 == 0:
                    fin_count += 1
                    sum_angle = angle[i]
                    fin_angle = self.angle_0[fin_count-1]
                else:
                    sum_angle += angle[i]
                hol = self.fin_length[i+fin_count] * np.cos(sum_angle)
                self.node_point[0, i+fin_count+1] = hol * np.cos(fin_angle) + self.node_point[0, i+fin_count]
                self.node_point[1, i+fin_count+1] = hol * np.sin(fin_angle) + self.node_point[1, i+fin_count]
                self.node_point[2, i+fin_count+1] = self.fin_length[i+fin_count] * np.sin(sum_angle) + self.node_point[2, i+fin_count]
        key = getkey()
        if key == 1792835:
            self.Sita_0 += np.pi / 18
            if self.Sita_0 > np.pi:
                self.Sita_0 = -np.pi
        if key == 1792836:
            self.Sita_0 -= np.pi / 18
            if self.Sita_0 < -np.pi:
                self.Sita_0 = np.pi
        if key == 1792833:
            self.Sita_1 += np.pi / 18
            if self.Sita_1 > np.pi / 2:
                self.Sita_1 = np.pi / 2
        if key == 1792834:
            self.Sita_1 -= np.pi / 18
            if self.Sita_1 < -np.pi / 2:
                self.Sita_1 = -np.pi / 2
        print('手を回転する：横方向', int(180*self.Sita_0/np.pi), '°  、縦方向', int(180*self.Sita_1/np.pi), '°')
        ROT_0 = np.array([[np.cos(self.Sita_0), 0, -np.sin(self.Sita_0)], [0, 1, 0], [np.sin(self.Sita_0), 0, np.cos(self.Sita_0)]])
        ROT_1 = np.array([[1, 0, 0], [0, np.cos(self.Sita_1), -np.sin(self.Sita_1)], [0, np.sin(self.Sita_1), np.cos(self.Sita_1)]])
        node_rot_point = ROT_0 @ ROT_1 @ self.node_point
        node_pixel = 100 * np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ node_rot_point + np.array([640*np.ones(self.num_node), 550*np.ones(self.num_node), np.zeros(self.num_node)])
        node_pixel = np.asarray(node_pixel, dtype=int)
        hand_draw = np.zeros((self.im_height, self.im_width, 3), dtype=np.uint8)
        for i in range(self.num_node):
            cv2.circle(hand_draw, (node_pixel[0, i], node_pixel[1, i]), 5, (0, 255, 0), -1)
        for i in range(self.num_node - 1):
            if i % 4 == 0:
                j_0 = 0
            else:
                j_0 = i
            j_1 = i + 1
            cv2.line(hand_draw, (node_pixel[0, j_0], node_pixel[1, j_0]), (node_pixel[0, j_1], node_pixel[1, j_1]), (0, 255, 0), 3)
        cv2.imshow('3D model', hand_draw)
        cv2.waitKey(1)


# mainという名前の関数です。C++のmain関数とは異なり、これは処理の開始地点ではありません。
def main(args=None):
    # rclpyの初期化処理です。ノードを立ち上げる前に実装する必要があります。
    rclpy.init(args=args)
    # HandShowクラスのインスタンスを作成
    minimal_subscriber = HandShow()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(minimal_subscriber)
    # 明示的にノードの終了処理を行います。
    minimal_subscriber.destroy_node()
    # rclpyの終了処理、これがないと適切にノードが破棄されないため様々な不具合が起こります。
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

if __name__ == '__main__':
    # 関数`main`を実行する。
    main()
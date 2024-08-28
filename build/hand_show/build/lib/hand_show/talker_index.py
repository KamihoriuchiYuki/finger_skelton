import numpy as np
#import rospy
import rclpy
from rclpy.node import Node
import fcntl
import termios
import sys
import os
from std_msgs.msg import Float64MultiArray #int型のメッセージのimport

class RsPub(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(Float64MultiArray, '/predicted_output_topic', 10)
        timer_period = 0.01  # seconds
        # タイマーを作成、一定時間ごとにコールバックを実行できるように設定
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.mode = 3
        self.num_input = 15
        self.angle = np.zeros((1, self.num_input))
        self.angle[0, 0] = 7 * np.pi / 36
        self.angle[0, 1] = np.pi / 2
        self.angle[0, 2] = np.pi / 2
        self.angle[0, 6] = 5 * np.pi / 12
        self.angle[0, 7] = np.pi / 2
        self.angle[0, 8] = 5 * np.pi / 12
        self.angle[0, 9] = 5 * np.pi / 12
        self.angle[0, 10] = np.pi / 2
        self.angle[0, 11] = 5 * np.pi / 12
        self.angle[0, 12] = 5 * np.pi / 12
        self.angle[0, 13] = np.pi / 2
        self.angle[0, 14] = 5 * np.pi / 12

    def timer_callback(self):
        key = getkey()
        #up:1792833
        #down:1792834
        #right:1792835
        #left:1792836
        if key == 1792835:
            self.mode += 1
            if self.mode == 6:
                self.mode = 3
        if key == 1792836:
            self.mode -= 1
            if self.mode == 2:
                self.mode = 5
        if key == 1792833:
            self.angle[0, self.mode] += np.pi / 36
            if self.angle[0, self.mode] > np.pi / 2:
                self.angle[0, self.mode] = np.pi / 2
        if key == 1792834:
            self.angle[0, self.mode] -= np.pi / 36
            if self.angle[0, self.mode] < 0.0:
                self.angle[0, self.mode] = 0.0
        fin_num = 3 - self.mode % 3
        finger = int((self.mode - self.mode % 3) / 3)
        if finger == 0:
            fin_name = '親指'
        if finger == 1:
            fin_name = '人差し指'
        if finger == 2:
            fin_name = '中指'
        if finger == 3:
            fin_name = '薬指'
        if finger == 4:
            fin_name = '小指'
        print(fin_name, 'の第', fin_num, '関節を', int(180*self.angle[0, self.mode]/np.pi), '°  曲げる')
        # 文字列型のインスタンスを作成します、この変数はローカル変数のため、コールバックの外からはアクセスできません。
        msg_data = Float64MultiArray() #データ格納用の型を作成
        msg_data.data = [] #データ格納配列の中身を毎回初期化
        for i in range(self.num_input):
            msg_data.data.append(self.angle[0, i])
        # Publisher経由でメッセージを発行
        self.publisher.publish(msg_data)


def main(args=None):
    # rclpyの初期化処理です。ノードを立ち上げる前に実装する必要があります。
    rclpy.init(args=args)
    # MinimalPublisherクラスのインスタンスを作成
    minimal_publisher = RsPub()
    # spin処理を実行、spinをしていないとROS 2のノードはデータを入出力することが出来ません。
    rclpy.spin(minimal_publisher)
    # 明示的にノードの終了処理を行います。
    minimal_publisher.destroy_node()
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

# 本スクリプト(publish.py)の処理の開始地点です。
if __name__ == '__main__':
    main()
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import fcntl
import termios
import sys
import os
from matplotlib import pyplot as plt

# MediaPipeの初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

num_node = 21
num_edge = 20

n_1 = 0
n_2 = 0

# RealSenseの初期化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

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

try:
    while True:
        # フレーム取得
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # フレームをnumpy配列に変換
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # MediaPipeによる手指の検出
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        finger = np.zeros((3, num_node))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = color_image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    #print(id)

                    # 画像の範囲内か確認
                    if cx < 0 or cx >= w or cy < 0 or cy >= h:
                        continue

                    # 深度情報の取得
                    depth = depth_image[cy, cx]

                    # カメラ座標系に変換
                    depth_scale = depth_frame.get_units()
                    depth_in_meters = depth * depth_scale
                    point = rs.rs2_deproject_pixel_to_point(
                        depth_frame.get_profile().as_video_stream_profile().get_intrinsics(), [cx, cy], depth_in_meters)

                    # pointは[x, y, z]の三次元座標
                    #print(f'Landmark {id}: x={point[0]}, y={point[1]}, z={point[2]}')
                    for i in range(3):
                        finger[i, id] = point[0]

        # 結果を表示
        cv2.imshow('RealSense', color_image)
        third_draw = np.zeros((480, 640, 3), dtype=np.uint8)
        image_shape = third_draw.shape
        key_0 = getkey()
        print(finger[:, 8])
        if key_0 == 1792833:
            if n_1 != 9:
                n_1 += 1
        if key_0 == 1792834:
            if n_1 != -9:
                n_1 -= 1
        if key_0 == 1792835:
            n_2 += 1
            if n_2 == 19:
                n_2 = -17
        if key_0 == 1792836:
            n_2 -= 1
            if n_2 == -18:
                n_2 = 18
        Sita_1 = n_1 * np.pi / 18
        Sita_2 = n_2 * np.pi / 18
        M_1 = np.array([[np.cos(Sita_2), 0, -np.sin(Sita_2)], [0, 1, 0], [np.sin(Sita_2), 0, np.cos(Sita_2)]])
        M_2 = np.array([[1, 0, 0], [0, np.cos(Sita_1), -np.sin(Sita_1)], [0, np.sin(Sita_1), np.cos(Sita_1)]])
        M = np.dot(M_2, M_1)
        finger_1 = np.dot(M, finger)
        nodes = np.zeros((num_node, 2), dtype =int)
        x_max = np.amax(finger_1[0, :])
        x_min = np.amin(finger_1[0, :])
        y_max = np.amax(finger_1[1, :])
        y_min = np.amin(finger_1[1, :])
        x_dif = x_max - x_min
        y_dif = y_max - y_min
        dif = max(x_dif, y_dif)
        O_point_x = (x_max + x_min) / 2
        O_point_y = (y_max + y_min) / 2
        for i in range(num_node):
            if dif == 0.0:
                x = 0.0
                y = 0.0
            else:
                x = (finger_1[0, i] - O_point_x) / dif
                y = (finger_1[1, i] - O_point_y) / dif
            x_pixel = x * image_shape[0]
            y_pixel = y * image_shape[0]
            nodes[i, 0] = int(x_pixel + 320)
            nodes[i, 1] = int(y_pixel + 240)
            #print(nodes)
            if x != 0.0:
                if y != 0.0:
                    cv2.circle(third_draw, (nodes[i, 0], nodes[i, 1]), 5, (0, 255, 0), -1)
        for i in range(num_edge):
            r = i % 4
            if r == 0:
                j_0 = 0
            else:
                j_0 = i
            j_1 = i + 1
            if finger_1[0, j_0] != 0.0:
                if finger_1[0, j_1] != 0.0:
                    cv2.line(third_draw, (nodes[j_0, 0], nodes[j_0, 1]), (nodes[j_1, 0], nodes[j_1, 1]), (0, 255, 0), 3)
        plt.imshow('3D model', third_draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # クリーンアップ
    pipeline.stop()
    cv2.destroyAllWindows()
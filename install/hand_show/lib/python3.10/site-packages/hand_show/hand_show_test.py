import cv2
import numpy as np
import fcntl
import termios
import sys
import os

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

mode = 0
mode_0 = 0
num_node = 21
num_fin = 5
num_input = 15
im_height = 640
im_width = 640
Sita_0 = 0.0
Sita_1 = 0.0
angle = np.zeros((1, num_input))

# 各指の長さを格納
fin_length = np.array([0.5, 1.0, 0.8, 0.6, 1.5, 1.0, 0.8, 0.5, 1.5, 1.0, 0.8, 0.6, 1.5, 1.0, 0.8, 0.5, 1.5, 0.8, 0.6, 0.5])

node_point = np.zeros((3, num_node))
angle_0 = np.array([np.pi / 9, np.pi * 5 / 12, np.pi / 2, np.pi * 7 / 12, np.pi * 2 / 3])
for i in range(num_fin):
    node_point[0, 4*i+1] = fin_length[4*i] * np.cos(angle_0[i])
    node_point[1, 4*i+1] = fin_length[4*i] * np.sin(angle_0[i])
while True:
    sum_angle = 0.0
    fin_count = 1
    for i in range(num_input):
        if i == 0:
            fin_angle = angle_0[0] + angle[0, 0]
            node_point[0, 2] = fin_length[1] * np.cos(fin_angle) + node_point[0, 1]
            node_point[1, 2] = fin_length[1] * np.sin(fin_angle) + node_point[1, 1]
        else:
            if i % 3 == 0:
                fin_count += 1
                sum_angle = angle[0, i]
                fin_angle = angle_0[fin_count-1]
            else:
                sum_angle += angle[0, i]
            hol = fin_length[i+fin_count] * np.cos(sum_angle)
            node_point[0, i+fin_count+1] = hol * np.cos(fin_angle) + node_point[0, i+fin_count]
            node_point[1, i+fin_count+1] = hol * np.sin(fin_angle) + node_point[1, i+fin_count]
            node_point[2, i+fin_count+1] = fin_length[i+fin_count] * np.sin(sum_angle) + node_point[2, i+fin_count]
    key = getkey()
    #up:1792833
    #down:1792834
    #right:1792835
    #left:1792836
    #c:99
    if key == 99:
        mode_0 += 1
        if mode_0 == 2:
            mode_0 = 0
    if key == 1792835:
        if mode_0 == 1:
            mode += 1
            if mode == num_input:
                mode = 0
        else:
            Sita_0 += np.pi / 18
            if Sita_0 > np.pi:
                Sita_0 = -np.pi
    if key == 1792836:
        if mode_0 == 1:
            mode -= 1
            if mode == -1:
                mode = num_input - 1
        else:
            Sita_0 -= np.pi / 18
            if Sita_0 < -np.pi:
                Sita_0 = np.pi
    if key == 1792833:
        if mode_0 == 1:
            angle[0, mode] += np.pi / 36
            if angle[0, mode] > np.pi / 2:
                angle[0, mode] = np.pi / 2
        else:
            Sita_1 += np.pi / 18
            if Sita_1 > np.pi / 2:
                Sita_1 = np.pi / 2
    if key == 1792834:
        if mode_0 == 1:
            angle[0, mode] -= np.pi / 36
            if angle[0, mode] < 0.0:
                angle[0, mode] = 0.0
        else:
            Sita_1 -= np.pi / 18
            if Sita_1 < -np.pi / 2:
                Sita_1 = -np.pi / 2
    if mode_0 == 0:
        print('手を回転する：横方向', int(180*Sita_0/np.pi), '°  、縦方向', int(180*Sita_1/np.pi), '°')
    if mode_0 == 1:
        fin_num = 3 - mode % 3
        finger = int((mode - mode % 3) / 3)
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
        print(fin_name, 'の第', fin_num, '関節を', int(180*angle[0, mode]/np.pi), '°  曲げる')
    ROT_0 = np.array([[np.cos(Sita_0), 0, -np.sin(Sita_0)], [0, 1, 0], [np.sin(Sita_0), 0, np.cos(Sita_0)]])
    ROT_1 = np.array([[1, 0, 0], [0, np.cos(Sita_1), -np.sin(Sita_1)], [0, np.sin(Sita_1), np.cos(Sita_1)]])
    node_rot_point = ROT_0 @ ROT_1 @ node_point
    node_pixel = 100 * np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ node_rot_point + np.array([320*np.ones(num_node), 550*np.ones(num_node), np.zeros(num_node)])
    node_pixel = np.asarray(node_pixel, dtype=int)
    hand_draw = np.zeros((im_height, im_width, 3), dtype=np.uint8)
    for i in range(num_node):
        cv2.circle(hand_draw, (node_pixel[0, i], node_pixel[1, i]), 5, (0, 255, 0), -1)
    for i in range(num_node - 1):
        if i % 4 == 0:
            j_0 = 0
        else:
            j_0 = i
        j_1 = i + 1
        cv2.line(hand_draw, (node_pixel[0, j_0], node_pixel[1, j_0]), (node_pixel[0, j_1], node_pixel[1, j_1]), (0, 255, 0), 3)
    cv2.imshow('3D model', hand_draw)
    cv2.waitKey(1)

## 説明
# cキーを入力することでモードチェンジ
# モード１：手全体を回転する
#     右キー：右回転
#     左キー：左回転
#     上キー：上回転
#     下キー：下回転
# モード２：各関節角度を変える
#     左右キー：角度を変える関節を変更する
#     上下キー：選択された関節の角度を変える
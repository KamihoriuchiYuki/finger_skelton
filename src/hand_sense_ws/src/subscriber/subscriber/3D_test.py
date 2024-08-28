import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

# MediaPipeのHandモデルを読み込む
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# RealSenseカメラの初期化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

vis = o3d.visualization.Visualizer()
vis.create_window('3D Point Cloud', width=1280, height=720)

# カメラのキャリブレーションパラメータ (例として使用)
camera_matrix = np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]])  # 仮のカメラ行列

# 手のポイントクラウドを初期化
point_cloud = o3d.geometry.PointCloud()

# 関節間の線分を定義
connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
               (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), 
               (0, 17), (17, 18), (18, 19), (19, 20)]

while True:
    # RealSenseカメラからフレームを取得
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue
    
    # 2D画像に手の関節位置を表示するための処理
    image = np.asanyarray(color_frame.get_data())
    image_height, image_width, _ = image.shape

    # MediaPipeを使って手の検出を行う
    results = hands.process(cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        # 検出された手の座標を取得
        for hand_landmarks in results.multi_hand_landmarks:
            hand_points = []
            for landmark in hand_landmarks.landmark:
                # 2D座標を取得
                image_x, image_y = int(landmark.x * color_frame.get_width()), int(landmark.y * color_frame.get_height())
                cv2.circle(image, (image_x, image_y), 5, (0, 255, 0), -1)

                # 3D座標を取得
                depth_x, depth_y = int(landmark.x * depth_frame.get_width()), int(landmark.y * depth_frame.get_height())
        
                # depth_x と depth_y の値を範囲内に制限する
                depth_x = min(max(depth_x, 0), depth_frame.get_width() - 1)
                depth_y = min(max(depth_y, 0), depth_frame.get_height() - 1)
        
                depth = depth_frame.get_distance(depth_x, depth_y)
        
                # カメラ画像の2D座標から3D座標への変換
                image_point = np.array([[image_x], [image_y], [1]])
                depth_point = depth * np.linalg.inv(camera_matrix) @ image_point
                hand_points.append(depth_point.flatten())

            # 手の中心を計算
            hand_center = np.mean(hand_points, axis=0)
            
            # 手のポイントを手の中心を基準に配置
            hand_points_relative = 1000 * (hand_points - hand_center)
            print(hand_points_relative)

            # 手のポイントクラウドを更新
            point_cloud.clear()
            point_cloud.points = o3d.utility.Vector3dVector(hand_points_relative)

            # 関節間の線分を描画
            lines = []
            for connection in connections:
                point1 = hand_points_relative[connection[0]]
                point2 = hand_points_relative[connection[1]]
                lines.append([point1, point2])
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(hand_points_relative),
                lines=o3d.utility.Vector2iVector(connections),
            )

            # ポイントクラウドと線分を描画
            vis.clear_geometries()
            vis.add_geometry(point_cloud)
            vis.add_geometry(line_set)
            vis.poll_events()
            vis.update_renderer()
    
    # 2D画像に関節位置を表示
    cv2.imshow('Hand Joints', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラを停止
pipeline.stop()
cv2.destroyAllWindows()
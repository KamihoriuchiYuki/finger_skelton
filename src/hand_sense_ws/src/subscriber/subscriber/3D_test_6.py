import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# RealSenseパイプラインを設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# ストリームを開始
pipeline.start(config)

# アライメントオブジェクトを作成（深度フレームとカラー画像を同期させるため）
align_to = rs.stream.color
align = rs.align(align_to)

try:
    # フレームを取得
    for _ in range(30):  # 安定したフレームを得るために初期フレームをスキップ
        frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        raise Exception("Could not get frames")

    # 深度データとカラー画像をnumpy配列に変換
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # 深度データを2次元画像として表示
    plt.figure()
    plt.imshow(depth_image * 100, cmap='gray')
    plt.title('Depth Image')
    plt.colorbar()
    plt.show()

    # 深度画像をポイントクラウドに変換
    pc = rs.pointcloud()
    pc.map_to(color_frame)  # ポイントクラウドをカラー画像にマッピング
    points = pc.calculate(depth_frame)
    vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

    # カラー情報を取得
    tex_coords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    colors = np.zeros((vertices.shape[0], 3), dtype=np.uint8)
    for i in range(vertices.shape[0]):
        u, v = tex_coords[i]
        u = int(u * color_image.shape[1])
        v = int(v * color_image.shape[0])
        if 0 <= u < color_image.shape[1] and 0 <= v < color_image.shape[0]:
            colors[i] = color_image[v, u]
    
    colors = colors[:, ::-1]

    # 深度データをフィルタリングして、50cm以内のポイントを削除
    min_distance = 0.1  # 最小距離（メートル）
    max_distance = 0.5  # 最大距離（メートル）
    valid_indices = np.where((vertices[:, 2] > min_distance) & (vertices[:, 2] < max_distance))[0]
    vertices = vertices[valid_indices]
    colors = colors[valid_indices]

    # Open3DのPointCloudオブジェクトを作成
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # ポイントクラウドを保存
    o3d.io.write_point_cloud("output.ply", point_cloud)

finally:
    # パイプラインを停止
    pipeline.stop()

print("Point cloud saved to 'output.ply'")

# ポイントクラウドを読み込む
point_cloud = o3d.io.read_point_cloud("output.ply")

# ポイントクラウドを可視化する
o3d.visualization.draw_geometries([point_cloud])

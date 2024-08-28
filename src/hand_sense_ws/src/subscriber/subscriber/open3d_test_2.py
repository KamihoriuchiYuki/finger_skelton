import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import datetime

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
align = rs.align(rs.stream.color)

vis = o3d.visualization.Visualizer()
vis.create_window('3D Point Cloud', width=1280, height=720)
pointcloud = o3d.geometry.PointCloud()

try:
    while True:
        dt0 = datetime.datetime.now()
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Create point cloud
        depth_image = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
        color_image = o3d.geometry.Image(np.asanyarray(color_frame.get_data()))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image)
        
        intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

        # Transform point cloud for proper visualization
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pointcloud.points = pcd.points
        pointcloud.colors = pcd.colors

        # Visualize
        if not vis.add_geometry(pointcloud):
            vis.add_geometry(pointcloud)
        vis.update_geometry(pointcloud)
        vis.poll_events()
        vis.update_renderer()

        # Calculate and print FPS
        process_time = datetime.datetime.now() - dt0
        print("FPS: " + str(1 / process_time.total_seconds()))

except KeyboardInterrupt:
    print("\nProgram interrupted by user. Closing...")

finally:
    # Clean up
    pipeline.stop()
    vis.destroy_window()
    del vis
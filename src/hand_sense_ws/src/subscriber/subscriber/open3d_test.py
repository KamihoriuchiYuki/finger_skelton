import open3d #ライブラリのインポート

pointcloud = open3d.io.read_point_cloud("bun.pcd") #点群ファイルを変数pointcloudに保存
open3d.visualization.draw_geometries([pointcloud]) #点群を画像として表示
open3d.io.write_point_cloud("output.pcd", pointcloud) #output.pcbという名前の点群ファイルを出力
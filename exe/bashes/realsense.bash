#!/bin/bash
tab="gnome-terminal --tab -t"

#本来は以下の引数はlaunch fileに書いて実行する
sleep 0.1s; $tab "rs_cam" -- ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true
sleep 0.1s; $tab "rs_view" -- ros2 run subscriber angle_finger
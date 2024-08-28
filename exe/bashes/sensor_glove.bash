#!/bin/bash
tab="gnome-terminal --tab -t"

#本来は以下の引数はlaunch fileに書いて実行する
sleep 0.1s; $tab "sensor_glove" -- ros2 run ros2serial_arduino serial_receive_node 

#!/bin/bash
tab="gnome-terminal --tab -t"

#本来は以下の引数はlaunch fileに書いて実行する
# sleep 0.1s; $tab "data_handler" -- ros2 run data_handler topics_subscriber && python3 ~/Sensor-Glove/src/lstm_ae/src/check_data.py
sleep 0.1s; $tab "data_handler" -- bash -c "ros2 run data_handler topics_subscriber ; python3 ~/Sensor-Glove/src/lstm_ae/src/check_data.py" 

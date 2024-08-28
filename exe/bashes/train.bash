#!/bin/bash
tab="gnome-terminal --tab -t"

#本来は以下の引数はlaunch fileに書いて実行する
sleep 0.1s; $tab "train" -- bash -c "python3 $ros2_ws/src/lstm_ae/src/main.py; exec bash"
# sleep 0.1s; $tab "train" -- bash -c "python3 $ros2_ws/src/lstm_ae/src/main.py; echo 'Press [ENTER] to exit...'; read"
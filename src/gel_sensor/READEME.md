# ROS2とArduinoUno3でSerial通信
<参考>
https://qiita.com/tomoswifty/items/50d4a7af1e8c401e2610

# ArduinoR3から取得したデータをtopicへ(Serial.Writeを利用する)
ros2serial_arduino/src/serial_receive_node.cpp　//の内容を変更する
serial.writeを利用するならstd::stringではだめ

std_msgs::msg::Int16  //includeするファイルを変更する
#include "std_msgs/msg/int16.hpp"  //これをincludeする



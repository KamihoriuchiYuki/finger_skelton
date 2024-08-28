#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/int16_multi_array.hpp"

#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>

int openSerial(const char *device_name){
    int  fd1 = open(device_name, O_RDWR | O_NOCTTY | O_NONBLOCK);
    
    fcntl(fd1, F_SETFL, 0);
    struct termios conf_tio;
    tcgetattr(fd1, &conf_tio);
    
    speed_t BAUDRATE = B19200;
    cfsetispeed(&conf_tio, BAUDRATE);
    cfsetospeed(&conf_tio, BAUDRATE);

    conf_tio.c_lflag &= ~(ECHO | ICANON);

    conf_tio.c_cc[VMIN] = 0;
    conf_tio.c_cc[VTIME] = 0;

    tcsetattr(fd1, TCSANOW, &conf_tio);
    return fd1;
}

int main(int argc, char **argv){
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("serial_receive_node");
    auto serial_pub = node->create_publisher<std_msgs::msg::Int16MultiArray>("serial_in", 1000);

    char device_name[] = "/dev/ttyACM0";  
    int fd1 = openSerial(device_name);
    if(fd1 < 0){
        RCLCPP_ERROR(node->get_logger(), "Serial Failed: could not open %s", device_name);
        printf("Serial Failed \n");
        rclcpp::shutdown();
    }

    rclcpp::WallRate loop_rate(20);
    while(rclcpp::ok()){
        uint8_t buf[256] = {0};
        // int data;
        int flag = 0;
        std_msgs::msg::Int16MultiArray serial_msg;
        serial_msg.data.resize(5); // 5つのセンサー値を格納するためにリサイズ

        while(true){
            int recv_data = read(fd1, buf, sizeof(buf));
            if(recv_data > 0){
                flag = 1;
                // auto serial_msg = std::make_unique<std_msgs::msg::Int16>();
                // serial_msg->data = data;
                // serial_pub->publish(std::move(serial_msg));
                // d ata += std::string(buf, recv_data);

                // auto serial_msg = std::make_unique<std_msgs::msg::String>();
                // serial_msg->data = data;
                // serial_pub->publish(std::move(serial_msg));
                // std::cout << "recv: " << std::endl;
                //int tmp;
                // auto serial_msg = std::make_unique<std_msgs::msg::Int16>();
                for(int i = 0; i < recv_data && i/2 < 5; i += 2){
                    int data = ((buf[i] & 0xFF) | (buf[i+1] << 8));
                    printf("decoded :%d\n", data);
                    serial_msg.data[i/2] = data;  // 各センサー値を格納
                }
                serial_pub->publish(serial_msg);
                break;
                // auto serial_msg = std::make_unique<std_msgs::msg::Int16>();
                // serial_msg->data = data;
                // serial_pub->publish(std::move(serial_msg));
                // RCLCPP_INFO(node->get_logger(), "recv: %s", data.c_str());


            }else{
                if(flag == 0) break;
            }
        }
        loop_rate.sleep();
    }
    rclcpp::shutdown();
    return 0;
}
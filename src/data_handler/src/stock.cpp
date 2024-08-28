#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/int16_multi_array.hpp"

class DualSubscriberNode : public rclcpp::Node
{
public:
  DualSubscriberNode()
  : Node("dual_subscriber_node")
  {
    // トピック "finger_angle_topic" をサブスクライブ (Float64MultiArray型)
    subscription_1_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "finger_angle_topic", 10, std::bind(&DualSubscriberNode::topic1_callback, this, std::placeholders::_1));

    // トピック "serial_in" をサブスクライブ (Int16MultiArray型)
    subscription_2_ = this->create_subscription<std_msgs::msg::Int16MultiArray>(
      "serial_in", 10, std::bind(&DualSubscriberNode::topic2_callback, this, std::placeholders::_1));
  }

private:
  // トピック "finger_angle_topic" のコールバック関数
  void topic1_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) const
  {
    std::cout << "Received from finger_angle_topic: [";
    for (size_t i = 0; i < msg->data.size(); ++i)
    {
      std::cout << msg->data[i];
      if (i < msg->data.size() - 1)
      {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  // トピック "serial_in" のコールバック関数 (Int16MultiArray型)
  void topic2_callback(const std_msgs::msg::Int16MultiArray::SharedPtr msg) const
  {
    std::cout << "Received from serial_in: [";
    for (size_t i = 0; i < msg->data.size(); ++i)
    {
      std::cout << msg->data[i];
      if (i < msg->data.size() - 1)
      {
        std::cout << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_1_;
  rclcpp::Subscription<std_msgs::msg::Int16MultiArray>::SharedPtr subscription_2_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DualSubscriberNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

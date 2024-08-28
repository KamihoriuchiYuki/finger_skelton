#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/int16_multi_array.hpp"
#include <vector>

class CombineSubscriberNode : public rclcpp::Node
{
public:
  CombineSubscriberNode()
  : Node("combine_subscriber_node")
  {
    subscription_1_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "topic1", 10, std::bind(&CombineSubscriberNode::topic1_callback, this, std::placeholders::_1));

    subscription_2_ = this->create_subscription<std_msgs::msg::Int16MultiArray>(
      "topic2", 10, std::bind(&CombineSubscriberNode::topic2_callback, this, std::placeholders::_1));

    // タイマーを設定して、一定周期でデータを処理
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),  // 100msごとに実行
      std::bind(&CombineSubscriberNode::process_data, this));
  }

private:
  void topic1_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    latest_data_1_ = msg->data;  // 最新のデータを保存
  }

  void topic2_callback(const std_msgs::msg::Int16MultiArray::SharedPtr msg)
  {
    latest_data_2_ = msg->data;  // 最新のデータを保存
  }

  void process_data()
  {
    if (!latest_data_1_.empty() && !latest_data_2_.empty())
    {
      std::vector<double> combined_data;
      combined_data.insert(combined_data.end(), latest_data_1_.begin(), latest_data_1_.end());
      combined_data.insert(combined_data.end(), latest_data_2_.begin(), latest_data_2_.end());

      // ここでcombined_dataを処理するか、次のステップに渡します
      RCLCPP_INFO(this->get_logger(), "Combined data size: %zu", combined_data.size());
      // combined_dataを使って、他の処理やパブリッシュを行います
    }
  }

  std::vector<double> latest_data_1_;
  std::vector<int16_t> latest_data_2_;

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_1_;
  rclcpp::Subscription<std_msgs::msg::Int16MultiArray>::SharedPtr subscription_2_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CombineSubscriberNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

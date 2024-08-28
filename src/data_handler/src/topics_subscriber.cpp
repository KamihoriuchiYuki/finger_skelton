#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/int16_multi_array.hpp"
#include <vector>
#include <fstream>
#include <thread>
#include <atomic>
#include <termios.h>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>

class CombineSubscriberNode : public rclcpp::Node
{
public:
  CombineSubscriberNode()
  : Node("combine_subscriber_node"), save_enabled_(false)
  {
    // ディレクトリを作成（存在しない場合）
    std::string dir = std::string(std::getenv("HOME")) + "/Sensor-Glove/src/data_handler/data";
    mkdir(dir.c_str(), 0777);

    // 現在の時刻を取得して、ファイル名を作成
    auto t = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(t);
    std::stringstream ss;
    ss << dir << std::put_time(std::localtime(&now_c), "/%m%d_%H%M") << ".csv";
    // ss << dir << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S") << ".csv";
    csv_file_.open(ss.str());

    publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/combined_raw", 10);

    subscription_1_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/finger_angle_topic", 10, std::bind(&CombineSubscriberNode::realsense_callback, this, std::placeholders::_1));

    subscription_2_ = this->create_subscription<std_msgs::msg::Int16MultiArray>(
      "/serial_in", 10, std::bind(&CombineSubscriberNode::sensor_glove_callback, this, std::placeholders::_1));

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&CombineSubscriberNode::process_data, this));
    
    start_time = this->now(); //start_time変数をpublicとして宣言

    // CSVヘッダを書き込む
    csv_file_ << "t";
    for (int i = 0; i < 15; ++i) {
        csv_file_ << ",rs" << i;
    }
    for (int i = 0; i < 5; ++i) {
        csv_file_ << ",sg" << i;
    }
    csv_file_ << "\n"; // ヘッダ行を終了

    // キーボード入力を監視するスレッドを開始
    input_thread_ = std::thread([this]() { this->keyboard_input_listener(); });
  }

  ~CombineSubscriberNode()
  {
    if (csv_file_.is_open()) {
      csv_file_.close();
    }
    input_thread_.join();
  }

private:
  void realsense_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (!msg->data.empty()) {
      realsense_data = msg->data;
    }
  }

  void sensor_glove_callback(const std_msgs::msg::Int16MultiArray::SharedPtr msg)
  {
    if (!msg->data.empty()) {
      sensorglove_data = msg->data;
    }
  }

  void process_data()
  {
    if (!realsense_data.empty() && !sensorglove_data.empty())
    {
      if (save_enabled_)
      {
        auto now = this->now();
        double current_time = now.seconds() - start_time.seconds();

        std_msgs::msg::Float64MultiArray msg;
        size_t total_size = 1 + realsense_data.size() + sensorglove_data.size();
        msg.data.resize(total_size);

        csv_file_ << current_time;
        msg.data[0] = current_time;

        for (size_t i = 0; i < 15; ++i)
        {
            double x = (i < realsense_data.size()) ? realsense_data[i] : 0.0;
            csv_file_ << "," << x;
            msg.data[1+i] = x;
        }
        for (size_t i = 0; i < 5; ++i)
        {
            int16_t y = (i < sensorglove_data.size()) ? sensorglove_data[i] : 0;
            csv_file_ << "," << y;
            msg.data[1 + realsense_data.size() + i] = static_cast<double>(y);
        }
        csv_file_ << "\n"; // データ行を終了

        csv_file_.flush();
        publisher_->publish(msg);
      }

      realsense_data.clear();
      sensorglove_data.clear();
    }
  }

  void keyboard_input_listener()
  {
    // ターミナル入力を非カノニカルモードで設定
    struct termios oldt, newt;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);

    while (rclcpp::ok())
    {
      char c = getchar();
      if (c == 's')  // 's'キーを押すと保存のオンオフを切り替える
      {
        save_enabled_ = !save_enabled_;
        if(save_enabled_){
          start_time = this->now();
        } 
        RCLCPP_INFO(this->get_logger(), "Saving is now %s", save_enabled_ ? "enabled" : "disabled");
      }
    }

    // 設定を元に戻す
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  }

  std::vector<double> realsense_data;
  std::vector<int16_t> sensorglove_data;
  std::ofstream csv_file_;
  std::thread input_thread_;
  std::atomic<bool> save_enabled_;

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_1_;
  rclcpp::Subscription<std_msgs::msg::Int16MultiArray>::SharedPtr subscription_2_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time start_time ; //クラスメンバ変数として宣言
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CombineSubscriberNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

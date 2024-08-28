import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import csv
import os

class CSVPublisher(Node):
    def __init__(self):
        super().__init__('csv_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/combined_raw', 10)
        timer_period = 0.1  # パブリッシュの周期を1秒に設定
        self.timer = self.create_timer(timer_period, self.timer_callback)
        data_path = os.path.expanduser('~/Sensor-Glove/src/data_handler/data/0813_1559_ind_sample.csv')
        self.csv_data = self.read_csv_file(data_path)  # CSVファイルを読み込みます
        self.index = 0

    def read_csv_file(self, filename):
        data = []
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader) #skip header
            for row in csvreader:
                # 各行のデータをfloatに変換してリストに追加します
                data.append([float(x) for x in row])
        return data

    def timer_callback(self):
        if self.index < len(self.csv_data):
            msg = Float64MultiArray()
            msg.data = self.csv_data[self.index]
            self.publisher_.publish(msg)
            self.get_logger().info(f'Published: {msg.data}')
            self.index += 1
        else:
            self.get_logger().info('All data has been published.')
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    csv_publisher = CSVPublisher()
    rclpy.spin(csv_publisher)
    csv_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

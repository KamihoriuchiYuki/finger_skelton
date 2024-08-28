import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import os
import sensor_fusion.data_processor as dp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

class ModelPredictorNode(Node):
    def __init__(self):
        super().__init__('model_predictor_node')
        
        # Load the pre-trained model
        file_path = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/results/0817/1356")
        model_path = os.path.join(file_path, "lstm_ae_2i2o.keras")
        params = dp.read_json(os.path.join(file_path, "params.json"))
        self.timesteps = params["timesteps"]
        self.input_dim = params["input_dim"]
        self.model = load_model(model_path, compile=False)

        # Initialize buffer to store the latest [timesteps] number of data points
        self.buffer = np.zeros((self.timesteps, 2))

        # Initialize scalers (same as used during training)
        self.std_scaler_rs4 = StandardScaler()
        self.std_scaler_sg1 = StandardScaler()

        # Load and fit the scalers on the training data (assumed to be stored in params)
        df = dp.get_data(params['data_path'])
        df['rs4_filtered'] = df['rs4'].apply(lambda x: x if -90 <= x <= 150 else np.nan)
        df['sg1_filtered'] = df['sg1'].apply(lambda x: x if 100 <= x <= 600 else np.nan)
        self.std_scaler_rs4.fit(df['rs4_filtered'].values.reshape(-1, 1))
        self.std_scaler_sg1.fit(df['sg1_filtered'].values.reshape(-1, 1))

        # Subscriber to receive input data
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/combined_raw',
            self.listener_callback,
            10)

        # Publisher to send out predictions
        self.publisher_ = self.create_publisher(Float64MultiArray, 'predicted_angle', 10)

    def listener_callback(self, msg):
        # Convert incoming message to numpy array
        new_data = np.array(msg.data).reshape(1, -1)  # Assuming input is 1D
        self.get_logger().info(f'Received data: {new_data}')
        
        # Set values to NaN if they are outside the specified range
        filtered_rs4 = new_data[0, 0] if -90 <= new_data[0, 0] <= 150 else np.nan
        filtered_sg1 = new_data[0, 1] if 100 <= new_data[0, 1] <= 600 else np.nan

        # Replace NaNs with 0
        filtered_rs4 = np.nan_to_num(filtered_rs4)
        filtered_sg1 = np.nan_to_num(filtered_sg1)

        # Scale the filtered data
        scaled_rs4 = self.std_scaler_rs4.transform([[filtered_rs4]])[0, 0]
        scaled_sg1 = self.std_scaler_sg1.transform([[filtered_sg1]])[0, 0]

        # Update buffer with the new scaled data (append to the end, remove from the start)
        self.buffer = np.roll(self.buffer, shift=-1, axis=0)
        self.buffer[-1, :] = [scaled_rs4, scaled_sg1]

        if np.any(np.isnan(self.buffer)):
            self.get_logger().warn('Buffer contains NaNs, skipping prediction.')
            return

        if self.buffer.shape[0] >= self.timesteps:
            # Split buffer into two inputs for the model
            input_data_rs4 = self.buffer[:, 0].reshape(1, self.timesteps, 1)
            input_data_sg1 = self.buffer[:, 1].reshape(1, self.timesteps, 1)

            # Make prediction
            prediction_rs4, prediction_sg1 = self.model.predict([input_data_rs4, input_data_sg1], verbose=0)
            self.get_logger().info(f'Prediction RS4: {prediction_rs4}, Prediction SG1: {prediction_sg1}')

            # Rescale the last prediction value back to the original scale
            rs4_pred = self.std_scaler_rs4.inverse_transform(prediction_rs4[:, -1, 0].reshape(-1, 1)).flatten()
            sg1_pred = self.std_scaler_sg1.inverse_transform(prediction_sg1[:, -1, 0].reshape(-1, 1)).flatten()

            # Publish the last prediction as Float64MultiArray
            pred_msg = Float64MultiArray()
            pred_msg.data = [float(0), float(0), float(0), 
                             float(0), float(rs4_pred[0]), float(0), 
                             float(0), float(0), float(0), 
                             float(0), float(0), float(0), 
                             float(0), float(0), float(0)]  
            self.publisher_.publish(pred_msg)
            self.get_logger().info('Published last prediction as Float64MultiArray')

def main(args=None):
    rclpy.init(args=args)
    node = ModelPredictorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

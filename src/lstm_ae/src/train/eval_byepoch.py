import data_processor as dp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lstm_model
from sklearn.preprocessing import StandardScaler
import os
import glob
import time
import pickle
import data_handler as dh
from tensorflow.keras.models import load_model

# Get the path to the directory containing the model and checkpoints
file_path = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/results/0823/1913")


# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_220_350_1.csv" #0813_1559
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_220_350_2.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_300_400.csv"   #0810_1225
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_1.csv" #0812_1630_20min
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_2.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_4.csv"
data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_5.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_6.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_7.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_8.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_9.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_10.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_11.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_2.csv" #kami 1
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_6.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_7.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_9.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_10.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_11.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_12.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_14.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_450_550.csv"   #0813_1525
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_620_2.csv" #0817_1823_side
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_620_4.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_2.csv" #kami 3
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_5.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_6.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_8.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_10.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_11.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_12.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_1.csv" #kami 2
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_2.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_3.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_4.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_5.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_6.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_7.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_8.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_9.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_10.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_11.csv"
# data_path = "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_12.csv"

eval_data_path = os.path.expanduser(data_path)

# Get the parameters
param_path = os.path.join(file_path, "params.json")
params = dp.read_json(param_path)
timesteps = params["timesteps"]

# Load the base model (without weights)
model_path = os.path.join(file_path, "lstm_ae_2i2o.keras")
model = load_model(model_path, compile=False)
params['model_path'] = model_path
x_missing_rate = 0.4
y_missing_rate = 0.0
params['x_missing_rate'] = x_missing_rate
params['y_missing_rate'] = y_missing_rate
params['eval_data_path'] = eval_data_path

# Load the data
df = dp.get_data(eval_data_path)
df['rs4_filtered'] = df['rs4'].apply(lambda x: x if -90 <= x <= 130 else np.nan)
df['sg1_filtered'] = df['sg1'].apply(lambda x: x if 200 <= x <= 800 else np.nan)

# Load scalers from the first training
# std_scaler_rs4 = pickle.load(open(os.path.join(file_path, "std_scaler_rs4.pkl"), 'rb'))
# std_scaler_sg1 = pickle.load(open(os.path.join(file_path, "std_scaler_sg1.pkl"), 'rb'))
# data_handler = dh.DataHandler(eval_data_path, scaler_rs4=std_scaler_rs4, scaler_sg1=std_scaler_sg1)
data_handler = dh.DataHandler(eval_data_path, scaler_rs4=None, scaler_sg1=None)
std_scaler_rs4, std_scaler_sg1 = data_handler.get_scalers()

# Split data and introduce missingness
data_handler.split_data(test_size=0.5)
data_handler.introduce_missingness(x_missing_rate=0.3, y_missing_rate=0.0)

# Generate sequences for LSTM
X_eval, _, Y_eval, _, X_eval_missing, _, Y_eval_missing, _ = data_handler.make_sequence(timesteps, is_random=False)
_, X_eval, _, Y_eval, _, X_eval_missing, _, Y_eval_missing = data_handler.make_sequence(timesteps, is_random=False)
# Prepare zero-filled arrays
X_eval_zero = np.nan_to_num(X_eval, nan=0)
Y_eval_zero = np.nan_to_num(Y_eval, nan=0)

X_eval_missing_zero = np.nan_to_num(X_eval_missing, nan=0)
Y_eval_missing_zero = np.nan_to_num(Y_eval_missing, nan=0)

# Get all checkpoint files
checkpoint_path = os.path.join(file_path, "checkpoints")
checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_path, "weights_epoch_*.weights.h5")))
save_dir = dp.prep_save_dir()

# Inverse transform
X_eval_raw = std_scaler_rs4.inverse_transform(X_eval[:,-1,0].reshape(-1, 1)).flatten()
X_eval_missing = std_scaler_rs4.inverse_transform(X_eval_missing[:,-1,0].reshape(-1, 1)).flatten()

Y_eval_raw = std_scaler_sg1.inverse_transform(Y_eval[:,-1,0].reshape(-1, 1)).flatten()
Y_eval_missing = std_scaler_sg1.inverse_transform(Y_eval_missing[:,-1,0].reshape(-1, 1)).flatten()

# Plot original evaluation data (RS and SG with missing values)
eval_data = [X_eval_missing, Y_eval_raw]
eval_label = ['RS Eval (missing)', 'SG Eval (raw)']
dp.save_plot(eval_data, eval_label, save_dir, 'eval_data')

dp.save_params(params, save_dir, 'params')

# Iterate over each checkpoint and make predictions
for checkpoint in checkpoint_files:
    # Load weights from checkpoint
    model.load_weights(checkpoint)
    
    # Compile the model
    model.compile(optimizer='adam', loss=lstm_model.masked_mse_loss)

    # Make predictions
    X_hat_eval, Y_hat_eval = model.predict([X_eval_missing_zero, Y_eval_missing_zero])
    
    X_hat_eval = std_scaler_rs4.inverse_transform(X_hat_eval[:,-1,0].reshape(-1, 1)).flatten()
    Y_hat_eval = std_scaler_sg1.inverse_transform(Y_hat_eval[:,-1,0].reshape(-1, 1)).flatten()


    # Extract epoch number from the checkpoint file name
    epoch_number = os.path.basename(checkpoint).split('_')[-1].split('.')[0]

    # Evaluate RS values for evaluation data (Actual vs Predicted)
    evaluate_data_eval_rs = [X_eval_missing, X_hat_eval]
    evaluate_label_eval_rs = ['RS Actual Eval (missing)', 'RS Predicted Eval']
    dp.save_plot(evaluate_data_eval_rs, evaluate_label_eval_rs, save_dir, f'rs_missing_prediction_epoch_{epoch_number}')

    # Evaluate RS values for evaluation data (Actual vs Predicted)
    evaluate_data_eval_rs = [X_eval_raw, X_hat_eval]
    evaluate_label_eval_rs = ['RS Actual Eval (raw)', 'RS Predicted Eval']
    dp.save_plot(evaluate_data_eval_rs, evaluate_label_eval_rs, save_dir, f'rs_raw_prediction_epoch_{epoch_number}')

    # Evaluate SG values for evaluation data (Actual vs Predicted)
    evaluate_data_eval_sg = [Y_eval_raw, Y_hat_eval]
    evaluate_label_eval_sg = ['SG Actual Eval (raw)', 'SG Predicted Eval']
    dp.save_plot(evaluate_data_eval_sg, evaluate_label_eval_sg, save_dir, f'sg_prediction_epoch_{epoch_number}')

    print(f"Predictions for epoch {epoch_number} saved.")

# Optional beep sound when finished
for i in range(1):
    print("\a")
    time.sleep(1)

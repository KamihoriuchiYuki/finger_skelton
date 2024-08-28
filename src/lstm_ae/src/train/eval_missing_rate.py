import os
import data_processor as dp
import numpy as np
import pandas as pd
import data_handler as dh
from tensorflow.keras.models import load_model

# Define the base path
base_path = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/important_data/models")
save_dir = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/important_data/results")

# Define the evaluation data paths and corresponding filenames
# training data is p1g3_sg350
evaluation_data_paths = {
    "p1g1_sg480": "~/Sensor-Glove/src/data_handler/data/divided/sg_480_570_1.csv",
    "p1g2_sg600": "~/Sensor-Glove/src/data_handler/data/divided/sg_600_680_2.csv",
    "p1g3_sg350": "~/Sensor-Glove/src/data_handler/data/divided/sg_350_440_5.csv",  # 1728
    "p1g3_sg500": "~/Sensor-Glove/src/data_handler/data/divided/sg_500_620_4.csv",  # 1751
    "p2g1_sg350": "~/Sensor-Glove/src/data_handler/data/divided/sg_350_450_10.csv",  # 1744
    "p2g2_sg550": "~/Sensor-Glove/src/data_handler/data/divided/sg_550_700_5.csv",  # 1747
    "p2g3_sg500": "~/Sensor-Glove/src/data_handler/data/divided/sg_500_650_5.csv", 
}


# Define the model names
model_names = ["RS", "RS_missing", "RS_missing_SG", "RS_SG_both_missing"]

# Construct the model-specific directories using the base path and model names
model_dirs = [os.path.join(base_path, model_name) for model_name in model_names]

# Define the missing rates
missing_rates = [0.1, 0.3, 0.5]

# Iterate over each evaluation data path
for eval_name, data_path in evaluation_data_paths.items():
    eval_data_path = os.path.expanduser(data_path)
    
    for missing_rate in missing_rates:
        # Load the evaluation data
        df = dp.get_data(eval_data_path)
        df['rs4_filtered'] = df['rs4'].apply(lambda x: x if -90 <= x <= 130 else np.nan)
        df['sg1_filtered'] = df['sg1'].apply(lambda x: x if 200 <= x <= 800 else np.nan)

        # Initialize an empty DataFrame to store the common actual and missing data
        common_data_df = pd.DataFrame()

        # Process the first model to save the actual and missing data
        first_model_dir = model_dirs[0]
        param_path = os.path.join(first_model_dir, "params.json")
        params = dp.read_json(param_path)
        timesteps = params["timesteps"]

        # Load scalers from the first training
        data_handler = dh.DataHandler(eval_data_path, scaler_rs4=None, scaler_sg1=None)
        std_scaler_rs4, std_scaler_sg1 = data_handler.get_scalers()

        # Split data and introduce missingness with different rates
        data_handler.split_data(test_size=0.5)
        data_handler.introduce_missingness(x_missing_rate=missing_rate, y_missing_rate=0.0)

        # Generate sequences for LSTM
        X_eval, _, Y_eval, _, X_eval_missing, _, Y_eval_missing, _ = data_handler.make_sequence(timesteps, is_random=False)
        X_eval_missing_zero = np.nan_to_num(X_eval_missing, nan=0)
        Y_eval_missing_zero = np.nan_to_num(Y_eval_missing, nan=0)

        # Inverse transform
        X_eval_raw = std_scaler_rs4.inverse_transform(X_eval[:,-1,0].reshape(-1, 1)).flatten()
        X_eval_missing = std_scaler_rs4.inverse_transform(X_eval_missing[:,-1,0].reshape(-1, 1)).flatten()
        Y_eval_raw = std_scaler_sg1.inverse_transform(Y_eval[:,-1,0].reshape(-1, 1)).flatten()
        Y_eval_missing = std_scaler_sg1.inverse_transform(Y_eval_missing[:,-1,0].reshape(-1, 1)).flatten()

        # Save the common actual and missing data
        common_data_df['RS Actual Data'] = X_eval_raw
        common_data_df['RS Missing Data'] = X_eval_missing
        common_data_df['SG Actual Data'] = Y_eval_raw
        common_data_df['SG Missing Data'] = Y_eval_missing

        # Initialize an empty DataFrame to store all predictions
        all_models_data_df = common_data_df.copy()

        for model_name, model_dir in zip(model_names, model_dirs):
            # Load the model
            model_path = os.path.join(model_dir, "lstm_ae_2i2o.keras")  # Adjust if needed
            model = load_model(model_path, compile=False)

            # Make predictions using the model
            X_hat_eval, Y_hat_eval = model.predict([X_eval_missing_zero, Y_eval_missing_zero])
            
            X_hat_eval = std_scaler_rs4.inverse_transform(X_hat_eval[:,-1,0].reshape(-1, 1)).flatten()
            Y_hat_eval = std_scaler_sg1.inverse_transform(Y_hat_eval[:,-1,0].reshape(-1, 1)).flatten()

            # Append the predictions to the overall DataFrame
            all_models_data_df[f'RS Predicted Data - {model_name}'] = X_hat_eval
            all_models_data_df[f'SG Predicted Data - {model_name}'] = Y_hat_eval

            print(f"Predictions processed for model {model_name} with evaluation data {eval_name} and missing rate {missing_rate}.")

        # Save the combined DataFrame for this evaluation data and missing rate to a CSV file
        save_path = os.path.join(save_dir, f'{eval_name}_missing_rate_{missing_rate}.csv')
        all_models_data_df.to_csv(save_path, index=False)

        print(f"Combined predictions for all models with {eval_name} and missing rate {missing_rate} saved to {save_path}.")

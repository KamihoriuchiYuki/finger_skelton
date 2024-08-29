import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(data):
    # Separate features (XYZ coordinates and their changes) and labels (MCP, PIP, DIP, TIP)
    X = data[['Wrist_X', 'Wrist_Y', 'Wrist_Z',
              'MCP_X', 'MCP_Y', 'MCP_Z',
              'PIP_X', 'PIP_Y', 'PIP_Z',
              'DIP_X', 'DIP_Y', 'DIP_Z',
              'TIP_X', 'TIP_Y', 'TIP_Z']]
    
    # Calculate changes (differences between consecutive time steps)
    X_diff = X.diff().fillna(0)
    
    # Combine original features and their changes
    X_combined = pd.concat([X, X_diff], axis=1)
    
    # Targets
    y = data[['MCP', 'PIP', 'DIP', 'TIP']]
    
    return X_combined, y

# Train and update the model
def train_and_save_model(X, y, model_path):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize or load the model
    try:
        model = joblib.load(model_path)
        print("Model loaded from file.")
    except:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        print("New model initialized.")

    # Train the model on each target separately
    for joint in y.columns:
        model.fit(X_train, y_train[joint])
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test[joint], y_pred)
        print(f"Accuracy for {joint}: {accuracy:.2f}")

    # Save the model
    joblib.dump(model, model_path)
    print("Model saved.")

# Main function
def main():
    file_path = "learning_data/combined_data_20240829_163123.csv"
    model_path = "saved_model.joblib"
    
    data = load_data(file_path)
    X, y = preprocess_data(data)
    train_and_save_model(X, y, model_path)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Change keras to tensorflow.keras
from tensorflow.keras.layers import Input, LSTM, RepeatVector, concatenate, Dense, TimeDistributed, Dropout
from tensorflow.keras.models import Model

def lstm_predX(input_dim, latent_dim, timesteps):
    # encode
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=False)(inputs)
    encoded = Dropout(0.2)(encoded)

    # decode
    hidden = RepeatVector(timesteps)(encoded)
    decoded = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=False)(hidden)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(1)(decoded)

    # train
    model = Model(inputs, decoded)
    model.summary()
    return model

def lstm_predXY(input_dim, latent_dim, timesteps):
    # encode
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=False)(inputs)
    encoded = Dropout(0.2)(encoded)

    # decode
    hidden = RepeatVector(timesteps)(encoded)
    decoded = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=False)(hidden)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(2)(decoded)

    # train
    model = Model(inputs, decoded)
    model.summary()
    return model

def lstm_ae(input_dim, latent_dim, timesteps):
    # encode
    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=False)(inputs)
    encoded = Dropout(0.2)(encoded)

    # decode
    hidden = RepeatVector(timesteps)(encoded)
    decoded = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=True)(hidden)
    decoded = Dropout(0.2)(decoded)
    decoded = TimeDistributed(Dense(input_dim))(decoded)

    # train
    model = Model(inputs, decoded)
    model.summary()
    return model

def lstm_ae_2i1o(input_dim, latent_dim, timesteps):
    # encode
    inputs_x = Input(shape=(timesteps, input_dim))
    encoded_x = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=False)(inputs_x)
    encoded_x = Dropout(0.2)(encoded_x)

    inputs_y = Input(shape=(timesteps, input_dim))
    encoded_y = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=False)(inputs_y)
    encoded_y = Dropout(0.2)(encoded_y)
    hidden = concatenate([encoded_x, encoded_y])
    hidden = Dense(latent_dim)(hidden)
    # decode
    hidden = RepeatVector(timesteps)(hidden)
    decoded = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=True)(hidden)
    decoded = Dropout(0.2)(decoded)
    decoded = TimeDistributed(Dense(input_dim))(decoded)

    # train
    model = Model([inputs_x, inputs_y], decoded)
    model.summary()
    return model

def lstm_ae_2i2o(input_dim, latent_dim, timesteps):
    # encode
    inputs_x = Input(shape=(timesteps, input_dim))
    encoded_x = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=False)(inputs_x)
    encoded_x = Dropout(0.2)(encoded_x)

    inputs_y = Input(shape=(timesteps, input_dim))
    encoded_y = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=False)(inputs_y)
    encoded_y = Dropout(0.2)(encoded_y)
    hidden = concatenate([encoded_x, encoded_y])
    hidden = Dense(round(latent_dim/2))(hidden)
    hidden_x = Dense(latent_dim)(hidden)
    hidden_y = Dense(latent_dim)(hidden)
    hidden_x = RepeatVector(timesteps)(hidden_x)
    hidden_y = RepeatVector(timesteps)(hidden_y)
    decoded_x = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=True)(hidden_x)
    decoded_y = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=True)(hidden_y)
    decoded_x = Dropout(0.2)(decoded_x)
    decoded_y = Dropout(0.2)(decoded_y)
    decoded_x = TimeDistributed(Dense(1))(decoded_x)
    decoded_y = TimeDistributed(Dense(1))(decoded_y)
    # # decode
    # hidden = RepeatVector(timesteps)(hidden)
    # decoded = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=True)(hidden)
    # decoded = Dropout(0.2)(decoded)
    # decoded_x = TimeDistributed(Dense(1))(decoded)
    # decoded_y = TimeDistributed(Dense(1))(decoded)
    
    # decoded_x = Dense(timesteps)(decoded)
    # decoded_y = Dense(timesteps)(decoded)
    model = Model([inputs_x, inputs_y], [decoded_x, decoded_y])
    model.summary()
    return model

def squared_difference_loss(y_pred, y_true):
    # y_pred [[0.1,0.8,....,(10クラス分の予測値)],[....][....]..(バッチサイズ個分)]
    # y_true [[1.0,0.0,0.0....,(正解のone-hot値)],[....][....]..(バッチサイズ個分)]
    print("y_pred.shape : ", y_pred.shape) #y_pred.shape :  (None, 100, 10)
    print("y_true.shape : ", y_true.shape) #y_true.shape :  (None, 100, 10)
    sd = tf.math.squared_difference(y_pred,y_true) #全ての要素ごとに差をとった２乗を演算

    rs = tf.math.reduce_sum(sd, axis=1) # 要素をデータごとに集計


    return rs

def masked_mse_loss(y_true, y_pred):
    print("y_pred.shape : ", y_pred.shape)  # e.g., y_pred.shape :  (None, 100, 10)
    print("y_true.shape : ", y_true.shape)  # e.g., y_true.shape :  (None, 100, 10)

    # Calculate the squared difference between predictions and true values
    sd = tf.math.squared_difference(y_pred, y_true)

    # Create a weight mask where y_true is not zero
    weight_mask = tf.cast(tf.math.not_equal(y_true, 0), dtype=tf.float32)

    # Apply the weight mask to the squared differences
    weighted_sd = tf.multiply(sd, weight_mask)

    # Sum the weighted squared differences across the last axis (classes)
    sum_weighted_sd = tf.math.reduce_sum(weighted_sd, axis=-1)

    # Count the number of non-zero elements in y_true (for weighting)
    non_zero_count = tf.math.reduce_sum(weight_mask, axis=-1)

    # Avoid division by zero by adding a small epsilon where non_zero_count is zero
    epsilon = tf.keras.backend.epsilon()
    non_zero_count = tf.maximum(non_zero_count, epsilon)

    # Compute the cost by dividing the sum of weighted squared differences by the number of non-zero elements
    cost = tf.math.divide(sum_weighted_sd, non_zero_count)

    # Optionally, you can further reduce the sum across the time axis (axis 1) if needed
    # cost = tf.math.reduce_sum(cost, axis=1)

    return cost


def masked_mse_loss_weight(y_true, y_pred):
    print("y_pred.shape : ", y_pred.shape)
    print("y_true.shape : ", y_true.shape)  

    # Calculate the squared difference between predictions and true values
    sd = tf.math.squared_difference(y_pred, y_true)

    # Create a weight mask where y_true is not zero
    weight_mask = tf.cast(tf.math.not_equal(y_true, 0), dtype=tf.float32)

    # Create a mask that doubles the weight for the last timestep
    # Here we assume that y_pred has the shape (batch_size, timesteps, 1)
    batch_size = tf.shape(y_pred)[0]
    timesteps = tf.shape(y_pred)[1]
    
    last_timestep_mask = tf.concat([tf.ones((batch_size, timesteps - 1, 1)), tf.ones((batch_size, 1, 1)) * 10], axis=1)

    # Apply the last timestep mask to the weight mask
    weight_mask = weight_mask * last_timestep_mask

    # Apply the weight mask to the squared differences
    weighted_sd = tf.multiply(sd, weight_mask)

    # Sum the weighted squared differences across the last axis (classes)
    sum_weighted_sd = tf.math.reduce_sum(weighted_sd, axis=-1)

    # Count the number of non-zero elements in y_true (for weighting)
    non_zero_count = tf.math.reduce_sum(weight_mask, axis=-1)

    # Avoid division by zero by adding a small epsilon where non_zero_count is zero
    epsilon = tf.keras.backend.epsilon()
    non_zero_count = tf.maximum(non_zero_count, epsilon)

    # Compute the cost by dividing the sum of weighted squared differences by the number of non-zero elements
    cost = tf.math.divide(sum_weighted_sd, non_zero_count)

    return cost
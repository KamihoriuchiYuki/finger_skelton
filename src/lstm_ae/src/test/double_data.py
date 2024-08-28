import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Change keras to tensorflow.keras
from tensorflow.keras.layers import Input, LSTM, RepeatVector, concatenate, Dense, TimeDistributed, Dropout
from tensorflow.keras.models import Model

# データを作成する関数
def make_data(random_factor, freq, duration, step, timesteps):
    def _load_data(data, n_prev = 100):  
        """
        data should be pd.DataFrame()
        """
        docX, docY = [], []
        for i in range(len(data)-n_prev):
            docX.append(data.iloc[i:i+n_prev].to_numpy())
        alsX = np.array(docX)

        return alsX

    def _train_test_split(df, test_size=0.1, n_prev = 100):  
        """
        This just splits data to training and testing parts
        """
        ntrn = round(len(df) * (1 - test_size))
        ntrn = int(ntrn)
        X_train = _load_data(df.iloc[0:ntrn], n_prev)
        X_test = _load_data(df.iloc[ntrn:], n_prev)

        return X_train, X_test

    np.random.seed(0)

    freq1, freq2 = freq
    t = np.arange(0, duration, step)
    df1 = pd.DataFrame(t, columns=["t"])
    df1["sin_t"] = df1.t.apply(lambda x: np.sin(x * (2 * np.pi * freq1)+ np.random.uniform(-1.0, +1.0) * random_factor))

    df2 = pd.DataFrame(t, columns=["t"])
    df2["sin_t2"] = df2.t.apply(lambda x: np.sin(x * (2 * np.pi * freq2)+ np.random.uniform(-1.0, +1.0) * random_factor))

    X_train, X_test = _train_test_split(df1[["sin_t"]], n_prev=timesteps) 
    Y_train, Y_test = _train_test_split(df2[["sin_t2"]], n_prev=timesteps) 

    # concatenate　X and make y
    # X_train = np.r_[X_train1, X_train2]
    # Y_train = np.r_[np.tile(0, X_train1.shape[0]), np.tile(1, X_train2.shape[0])]
    # X_test = np.r_[X_test1, X_test2]
    # Y_test = np.r_[np.tile(0, X_test1.shape[0]), np.tile(1, X_test2.shape[0])]


    # only use class 0
    # X_train = X_train1
    # Y_train = np.tile(0, X_train1.shape[0])
    # X_test = X_test1
    # Y_test = np.tile(0, X_test1.shape[0])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df1["t"].to_numpy(), df1["sin_t"].to_numpy() , label="freq " + str(freq1), color="red")
    ax.plot(df2["t"].to_numpy(), df2["sin_t2"].to_numpy(), label="freq " + str(freq2), color="blue")
    ax.legend(loc="upper right")
    fig.savefig("./sin_plot.png")
    plt.close("all")

    return X_train, Y_train, X_test, Y_test

# 乱数の係数
random_factor = 0.05
# 1サイクルの中で，何点サンプリングするか．
freq=(5,10)
duration = 10
step = 0.001
# windowの長さ．一つの系列の長さになる．
timesteps = 100
epochs = 150

X_train, Y_train, X_test, Y_test = make_data(random_factor, freq, duration, step, timesteps)
print("X_train.shape : ", X_train.shape) #X_train.shape :  (17802, 100, 1)
print("Y_train.shape : ", Y_train.shape) #Y_train.shape :  (17802,)
print("X_test.shape : ", X_test.shape) #X_test.shape :  (1800, 100, 1)
print("Y_test.shape : ", Y_test.shape) #Y_test.shape :  (1800,)

XY_train = np.concatenate([X_train, Y_train], axis=2)
XY_test = np.concatenate([X_test, Y_test], axis=2)

print("XY_train.shape : ", XY_train.shape) #XY_train.shape :
print("XY_test.shape : ", XY_test.shape) #XY_test.shape :  

input_dim = 2
latent_dim = 10

# encode
inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=False)(inputs)
encoded = Dropout(0.2)(encoded)
# #decode
hidden = RepeatVector(timesteps)(encoded)
# reverse_input = Input(shape=(timesteps, input_dim))
# hidden_revinput = concatenate([hidden, reverse_input])
# decoded = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=True)(hidden_revinput)
decoded = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=True)(hidden)
decoded = Dropout(0.2)(decoded)
# decoded = Dense(latent_dim, activation="relu")(decoded)
# decoded = Dense(input_dim, activation="tanh")(decoded)
decoded = TimeDistributed(Dense(input_dim))(decoded)

# train
LSTM_AE = Model(inputs, decoded)
# LSTM_AE = Model([inputs, reverse_input], decoded)
Model.summary(LSTM_AE)
LSTM_AE.compile(optimizer='rmsprop', loss='mse')
# X_train_rev = X_train[:,::-1,:]
# LSTM_AE.fit([X_train, X_train_rev], X_train, epochs=epochs, batch_size=300, shuffle=True, validation_data=([X_train, X_train_rev], X_train))

history = LSTM_AE.fit(XY_train, XY_train, epochs=epochs, batch_size=300, shuffle=True, validation_data=(XY_train, XY_train))
plt.plot(history.history['mse'], label='Train mse')
plt.plot(history.history['val_mse'], label='valid mse')
plt.ylabel('mse')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()



XY_hat = LSTM_AE.predict(XY_train)
XY_hat_test = LSTM_AE.predict(XY_test)
# X_hat = LSTM_AE.predict([X_train, X_train_rev])

def split_X(XY):
    X = XY[:,:,:1]
    Y = XY[:,:,1:]
    return X, Y

X_hat, Y_hat = split_X(XY_hat)
X_hat_test, Y_hat_test = split_X(XY_hat_test)
# print("X_train.shape : ", X_train.shape) #X_train.shape :  (3501, 100, 1)
# print("Y_train.shape : ", Y_train.shape) #Y_train.shape :  (14301, 100, 1)
print("X_hat.shape : ", X_hat.shape) #X_hat.shape :  (3501, 100, 1)
print("Y_hat.shape : ", Y_hat.shape) #Y_hat.shape :  (14301, 100, 1)

# reconstruct したX_trainがどんな感じか見てみる
def plot_save(start_index, X_hat, X_train, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in np.arange(start_index,start_index+5):
        #５個ずつプロットする．
        ax.plot(X_hat[i,:,0], label="X hat", color="red")
        ax.plot(X_train[i,:,0], label="X train", color="blue")
    savename = "./" + "double_" + str(epochs) + "ep/" + "AE_reconst_" + str(epochs) + "ep_start_" + str(start_index) + "_cls" + name + ".png"
    fig.savefig(savename)
    plt.close("all")

start_list = np.arange(0,len(X_train), 1000)
for start_index in start_list:
    plot_save(start_index, X_hat, X_train, name="X_train")

start_list = np.arange(0,len(Y_train), 1000)
for start_index in start_list:
    plot_save(start_index, Y_hat, Y_train, name="Y_train")

# reconstruct したX_testがどんな感じか見てみる
start_list = np.arange(0,len(X_test), 1000)
for start_index in start_list:
    plot_save(start_index, X_hat_test, X_test, name="X_test")

start_list = np.arange(0,len(Y_test), 1000)
for start_index in start_list:
    plot_save(start_index, Y_hat_test, Y_test, name="Y_test")


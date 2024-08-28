import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# データを作成する関数
def make_data(random_factor, number_of_cycles, \
                timesteps, sampling_num_pair):
    """
    sampling_num_pair : 2 elements of tupple
        1周期の間にサンプリングする数
        Ex. (20,80)
    """
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

    sampling_num1, sampling_num2 = sampling_num_pair

    df1 = pd.DataFrame(np.arange(sampling_num1 * number_of_cycles + 1), columns=["t"])
    df1["sin_t"] = df1.t.apply(lambda x: np.sin(x * (2 * np.pi / sampling_num1)+ np.random.uniform(-1.0, +1.0) * random_factor))

    df2 = pd.DataFrame(np.arange(sampling_num2 * number_of_cycles + 1), columns=["t"])
    df2["sin_t2"] = df2.t.apply(lambda x: np.sin(x * (2 * np.pi / sampling_num2)+ np.random.uniform(-1.0, +1.0) * random_factor))

    X_train1, X_test1 = _train_test_split(df1[["sin_t"]], n_prev=timesteps) 
    X_train2, X_test2 = _train_test_split(df2[["sin_t2"]], n_prev=timesteps) 

    # concatenate　X and make y
    X_train = np.r_[X_train1, X_train2]
    y_train = np.r_[np.tile(0, X_train1.shape[0]), np.tile(1, X_train2.shape[0])]

    X_test = np.r_[X_test1, X_test2]
    y_test = np.r_[np.tile(0, X_test1.shape[0]), np.tile(1, X_test2.shape[0])]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df1["sin_t"][0:80], label="class0 sampling_num 20", color="red")
    ax.plot(df2["sin_t2"][0:80], label="class1 sampling_num 80", color="blue")
    ax.legend(loc="upper right")
    fig.savefig("./sin_plot.png")
    plt.close("all")

    return X_train, y_train, X_test, y_test

# 乱数の係数
random_factor = 0.05
# 生成するサイクル数
number_of_cycles = 200
# 1サイクルの中で，何点サンプリングするか．
sampling_num_pair=(20,80)
# windowの長さ．一つの系列の長さになる．
timesteps = 100
epochs = 150

X_train, y_train, X_test, y_test = make_data(random_factor, number_of_cycles, timesteps, sampling_num_pair)
print("X_train.shape : ", X_train.shape) #X_train.shape :  (17802, 100, 1)
print("y_train.shape : ", y_train.shape) #y_train.shape :  (17802,)
print("X_test.shape : ", X_test.shape) #X_test.shape :  (1800, 100, 1)
print("y_test.shape : ", y_test.shape) #y_test.shape :  (1800,)

# LSTM Autoencoder
from keras.layers import Input, LSTM, RepeatVector, concatenate, Dense
from keras.models import Model

input_dim = 1
latent_dim = 10

# encode
inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=True)(inputs)

#decode
# hidden = RepeatVector(timesteps)(encoded)
# reverse_input = Input(shape=(timesteps, input_dim))
# hidden_revinput = concatenate([hidden, reverse_input])
decoded = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=True)(encoded)
# decoded = Dense(latent_dim, activation="relu")(decoded)
# decoded = Dense(input_dim, activation="tanh")(decoded)

# train
LSTM_AE = Model(inputs, decoded)
Model.summary(LSTM_AE)
LSTM_AE.compile(optimizer='rmsprop', loss='mse')
X_train_rev = X_train[:,::-1,:]
LSTM_AE.fit(X_train, X_train, epochs=epochs, batch_size=500, shuffle=True, validation_data=(X_train, X_train))

X_hat = LSTM_AE.predict(X_train)

def split_X(X, y):
    y_inv = np.abs(y - 1.)
    X_0 = X[y_inv.astype(bool),:,:]
    X_1 = X[y.astype(bool),:,:]
    return X_0, X_1

X_train_0, X_train_1 = split_X(X_train, y_train)
X_hat_0, X_hat_1 = split_X(X_hat, y_train)

print("X_train_0.shape : ", X_train_0.shape) #X_train_0.shape :  (3501, 100, 1)
print("X_train_1.shape : ", X_train_1.shape) #X_train_1.shape :  (14301, 100, 1)
print("X_hat_0.shape : ", X_hat_0.shape) #X_hat_0.shape :  (3501, 100, 1)
print("X_hat_1.shape : ", X_hat_1.shape) #X_hat_1.shape :  (14301, 100, 1)

# reconstruct したX_trainがどんな感じか見てみる
def plot_save(start_index, X_hat, X_train, X_class):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in np.arange(start_index,start_index+5):
        #５個ずつプロットする．
        ax.plot(X_hat[i,:,0], label="X hat", color="red")
        ax.plot(X_train[i,:,0], label="X train", color="blue")
    savename = "./" + str(epochs) + "ep/" + "AE_reconst_" + str(epochs) + "ep_start_" + str(start_index) + "_cls" + str(X_class) + ".png"
    fig.savefig(savename)
    plt.close("all")

start_list = np.arange(0,len(X_train_0), 1000)
for start_index in start_list:
    plot_save(start_index, X_hat_0, X_train_0, X_class=0)

start_list = np.arange(0,len(X_train_1), 1000)
for start_index in start_list:
    plot_save(start_index, X_hat_1, X_train_1, X_class=1)


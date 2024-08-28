import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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
    df2["sin_t2"] = df2.t.apply(lambda x: x * 0.05 + np.sin(x * (2 * np.pi * freq2)+ np.random.uniform(-1.0, +1.0) * random_factor))

    X_train, X_test = _train_test_split(df1[["sin_t"]], n_prev=timesteps) 
    Y_train, Y_test = _train_test_split(df2[["sin_t2"]], n_prev=timesteps) 

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(df1["t"].to_numpy(), df1["sin_t"].to_numpy() , label="freq " + str(freq1), color="red")
    ax.plot(df2["t"].to_numpy(), df2["sin_t2"].to_numpy(), label="freq " + str(freq2), color="blue")
    ax.legend(loc="upper right")
    fig.savefig("./sin_plot.png")
    plt.close("all")

    return X_train, Y_train, X_test, Y_test

def my_loss(y_pred, y_true):
    # y_pred [[0.1,0.8,....,(10クラス分の予測値)],[....][....]..(バッチサイズ個分)]
    # y_true [[1.0,0.0,0.0....,(正解のone-hot値)],[....][....]..(バッチサイズ個分)]
    print("y_pred.shape : ", y_pred.shape) #y_pred.shape :  (None, 100, 10)
    print("y_true.shape : ", y_true.shape) #y_true.shape :  (None, 100, 10)
    sd = tf.math.squared_difference(y_pred,y_true) #全ての要素ごとに差をとった２乗を演算

    rs = tf.math.reduce_sum(sd, axis=1) # 要素をデータごとに集計

    return rs



# 乱数の係数
random_factor = 0.05
# 1サイクルの中で，何点サンプリングするか．
freq=(5,10)
duration = 5
step = 0.001
# windowの長さ．一つの系列の長さになる．
timesteps = 50
epochs = 100

X_train, Y_train, X_test, Y_test = make_data(random_factor, freq, duration, step, timesteps)
print("X_train.shape : ", X_train.shape) #X_train.shape :  (17802, 100, 1)
print("Y_train.shape : ", Y_train.shape) #Y_train.shape :  (17802,)
print("X_test.shape : ", X_test.shape) #X_test.shape :  (1800, 100, 1)
print("Y_test.shape : ", Y_test.shape) #Y_test.shape :  (1800,)

XY_train = np.concatenate([X_train, Y_train], axis=2)
XY_test = np.concatenate([X_test, Y_test], axis=2)

XY_train_last = XY_train[:,-1,0]
XY_test_last = XY_test[:,-1,0]

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
decoded = LSTM(latent_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=False)(hidden)
decoded = Dropout(0.2)(decoded)
# decoded = Dense(latent_dim, activation="relu")(decoded)
decoded = Dense(1)(decoded)
# decoded = TimeDistributed(Dense(input_dim))(decoded)

# train
LSTM_AE = Model(inputs, decoded)
# LSTM_AE = Model([inputs, reverse_input], decoded)
Model.summary(LSTM_AE)
LSTM_AE.compile(optimizer='adam', loss=my_loss)
# X_train_rev = X_train[:,::-1,:]
# LSTM_AE.fit([X_train, X_train_rev], X_train, epochs=epochs, batch_size=300, shuffle=True, validation_data=([X_train, X_train_rev], X_train))

history = LSTM_AE.fit(XY_train, XY_train_last, epochs=epochs, batch_size=300, shuffle=True, validation_data=(XY_train, XY_train_last))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()



X_hat = LSTM_AE.predict(XY_train)
X_hat_test = LSTM_AE.predict(XY_test)
# X_hat = LSTM_AE.predict([X_train, X_train_rev])

def split_X(XY):
    X = XY[:,:,:1]
    Y = XY[:,:,1:]
    return X, Y

# X_hat, Y_hat = XY_hat[:,0], XY_hat[:,1]
# X_hat_test, Y_hat_test = XY_hat_test[:,0], XY_hat_test[:,1]
# print("X_train.shape : ", X_train.shape) #X_train.shape :  (3501, 100, 1)
# print("Y_train.shape : ", Y_train.shape) #Y_train.shape :  (14301, 100, 1)
print("X_hat.shape : ", X_hat.shape) #X_hat.shape :  (3501, 100, 1)
# print("Y_hat.shape : ", Y_hat.shape) #Y_hat.shape :  (14301, 100, 1)

# reconstruct したX_trainがどんな感じか見てみる
def plot_save(X_pred, X_true, name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X_pred, label="X hat", color="red")
    ax.plot(X_true[:,-1,0], label="X train", color="blue")
    savename = "./" + str(epochs) + "ep/" + "AE_reconst_" + str(epochs) + name + ".png"
    fig.savefig(savename)
    plt.close("all")

plot_save(X_hat, X_train, name="X_train")

# plot_save(Y_hat, Y_train, name="Y_train")

# reconstruct したX_testがどんな感じか見てみる
plot_save(X_hat_test, X_test, name="X_test")

# plot_save(Y_hat_test, Y_test, name="Y_test")


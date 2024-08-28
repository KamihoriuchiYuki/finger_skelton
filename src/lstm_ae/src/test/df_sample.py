import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

# make data frame from sin wave
freq1 = 20
freq2 = 40
duration = 1
step = 0.001
t = np.arange(0, duration, step)
df = pd.DataFrame(t, columns=["t"])
df["sin_t"] = df.t.apply(lambda x: np.sin(x * (2 * np.pi * freq1)))
df["sin_t2"] = df.t.apply(lambda x: np.sin(x * (2 * np.pi * freq2)))

X_train1, X_test1 = _train_test_split(df[["sin_t"]], n_prev=100) 
X_train2, X_test2 = _train_test_split(df[["sin_t2"]], n_prev=100) 
X_train = np.r_[X_train1, X_train2]
X_train = np.concatenate([X_train1, X_train2], axis=2)
print(X_train.shape)
#plot sin wave
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(df["t"].to_numpy(), df["sin_t"].to_numpy())
# fig.savefig("./sample_data_plot.png")
# plt.close("all")


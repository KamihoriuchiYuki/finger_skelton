import os
from datetime import datetime
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.signal import savgol_filter
from tensorflow.keras.utils import plot_model

# CSVファイルを読み込みDataFrameに変換する関数
def get_data(file_path):
    try:
        # pandasを使ってCSVファイルを読み込む
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print("指定されたファイルが見つかりません。ファイルパスを確認してください。")
        return None

def read_json(json_file_path):
    # Expand user path if necessary
    json_file_path = os.path.expanduser(json_file_path)
    
    # Load JSON data
    with open(json_file_path, 'r') as file:
        params = json.load(file)
    
    return params
# データを作成する関数
def make_sequence(data, n_prev = 100, random_add = None):  
    """
    data should be pd.DataFrame()
    """
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        # docX.append(data.iloc[i:i+n_prev].to_numpy())
        sequence = data.iloc[i:i+n_prev].to_numpy()
        # random_add = np.random.uniform(low=-1.0, high=1.0)  # ここでランダムな値を生成
        if random_add is not None:
            sequence += random_add[i]  # シーケンスにランダムな値を加算
        
        # random_add = np.random.uniform(low=-1.0, high=1.0)  # 足すためのランダムな値
        # random_multiply = np.random.uniform(low=0.5, high=1.5)  # 掛けるためのランダムな値
        # sequence = sequence* random_multiply  + random_add 

        docX.append(sequence)
    alsX = np.array(docX)

    return alsX

def split_data(df_t, df_data, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df_data) * (1 - test_size))
    ntrn = int(ntrn)
    # print(f"Total data points: {len(df_data)}, Training data points: {ntrn}, Test data points: {len(df_data) - ntrn}")
    # X_train = make_sequence(df_data.iloc[0:ntrn], n_prev)
    df_train = pd.DataFrame({
        "t": df_t.iloc[0:ntrn].to_numpy(),
        "data": df_data.iloc[0:ntrn].to_numpy()
    })

    df_test = pd.DataFrame({
        "t": df_t.iloc[ntrn:].to_numpy(),
        "data": df_data.iloc[ntrn:].to_numpy()
    })
    # df_test  = make_sequence(df.iloc[ntrn:], n_prev)
    print(f"df_train: {df_train.shape}, df_test: {df_test.shape}")

    return df_train, df_test

def make_dummy_data(random_factor, freq, amp, duration, step):
    np.random.seed(0)
    freq1, freq2 = freq
    amp1, amp2 = amp
    t = np.arange(0, duration, step)
    df = pd.DataFrame(t, columns=["t"])
    df["X"] = df.t.apply(lambda x:  amp1 * np.sin(x * (2 * np.pi * freq1)+ np.random.uniform(-1.0, +1.0) * random_factor))
    df["Y"] = df.t.apply(lambda x: x * 0.5 + amp2 * np.sin(x * (2 * np.pi * freq2)+ np.random.uniform(-1.0, +1.0) * random_factor))
    return df

def introduce_missingness(data, missing_rate):
    np.random.seed(42)  # 再現性のためにランダムシードを設定
    data_flat = data.values.flatten()
    # data_flat = data.flatten()
    n_total = data_flat.size
    n_missing = int(n_total * missing_rate)
    missing_indices = np.random.choice(n_total, n_missing, replace=False)
    data_flat[missing_indices] = np.nan
    data = data_flat.reshape(data.shape)
    return data

def get_missing_flags(data):
    data_flat = data.values.flatten()
    # data_flat = data.flatten()
    if_missing = np.isnan(data_flat)  # 欠損値のフラグを作成
    if_missing = if_missing.reshape(data.shape)
    
    return if_missing

# save
def save_plot(data_list, label_list, save_dir, savename, figsize=(25, 5)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for data, label in zip(data_list, label_list):
        ax.plot(data, label=label)
    ax.legend()
    save_path = os.path.join(save_dir, f"{savename}.png")
    fig.savefig(save_path)
    plt.close("all")
    print(f"Plot saved to: {save_path}")

def save_params(params, save_dir, filename):
    save_path = os.path.join(save_dir, f"{filename}.json")
    with open(save_path, "w") as f:
        json.dump(params, f, indent=4)
    print(f"Params saved to: {save_path}")

def save_model(model, save_dir, filename):
    model_save_path = os.path.join(save_dir, f"{filename}.keras")
    model.save(model_save_path)
    summary_save_path = os.path.join(save_dir, "model_summary.txt")
    with open(summary_save_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    plot_model(model, to_file=os.path.join(save_dir, 'model.png'), show_shapes=True)

    print(f"Model saved to: {model_save_path}")
    print(f"Model summary saved to: {summary_save_path}")

def _save_path(filename):
    # 現在の日時を取得
    now = datetime.now()
    # "MMDD/HHMM"の形式でフォルダを作成
    date_dir = now.strftime("%m%d")
    time_dir = now.strftime("%H%M")
    base_dir = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/results")
    # 保存するディレクトリパスを作成
    save_dir = os.path.join(base_dir, date_dir, time_dir)
    # ディレクトリが存在しない場合は作成
    os.makedirs(save_dir, exist_ok=True)
    # 保存するファイルのフルパス
    save_path = os.path.join(save_dir, filename)
    return save_path

def prep_save_dir():
    # 現在の日時を取得
    now = datetime.now()
    # "MMDD/HHMM"の形式でフォルダを作成
    date_dir = now.strftime("%m%d")
    time_dir = now.strftime("%H%M")
    base_dir = os.path.expanduser("~/Sensor-Glove/src/lstm_ae/results")
    # 保存するディレクトリパスを作成
    save_dir = os.path.join(base_dir, date_dir, time_dir)
    # ディレクトリが存在しない場合は作成
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def fft_lowpass(x, fmax, dt=1/20):
    # NaNを補完（今回は線形補間で補完）
    nan_mask = np.isnan(x)
    indices = np.arange(len(x))
    x_interp = np.interp(indices, indices[~nan_mask], x[~nan_mask])

    # FFTを適用
    freq_X = np.fft.fftfreq(len(x), dt)
    X_F = np.fft.fft(x_interp)

    # カットオフ周波数でフィルタリング
    X_F[(freq_X > fmax)] = 0
    X_F[(freq_X < -fmax)] = 0

    # 逆FFTを適用し、実数部分を取得
    x_CF = np.fft.ifft(X_F).real

    # 元のNaNの位置に再度NaNを適用
    x_CF[nan_mask] = np.nan

    return x_CF

def main_dummy():
    random_factor = 0.1
    freq = [1, 2]
    amp = [1, 1]    
    duration = 10
    step = 0.1
    timesteps = 2

    df = make_dummy_data(random_factor, freq, amp, duration, step)
    # print(df['X'].head())
    df_xtrain, df_xtest = split_data(df_t = df["t"], df_data=df['X'], test_size=0.1)
    df_ytrain, df_ytest = split_data(df_t = df["t"], df_data=df['Y'], test_size=0.1)
    
    missing_rate = 0.00
    df_xtrain['data_missing'] = introduce_missingness(df_xtrain['data'], missing_rate)
    df_xtrain['if_missing'] = df_xtrain['data_missing'].isnull() # label missing data
    df_xtest['data_missing']  = introduce_missingness(df_xtest['data'],  missing_rate)

    print(df.head())
    print(df_xtrain)
    print(df_xtest)

    X_train = make_sequence(df_xtrain['data_missing'], timesteps)
    X_test  = make_sequence(df_xtest['data_missing'],  timesteps)
    Y_train = make_sequence(df_ytrain['data'], timesteps)
    Y_test  = make_sequence(df_ytest['data'],  timesteps)
    # 次元を追加
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    Y_train = np.expand_dims(Y_train, axis=-1)
    Y_test = np.expand_dims(Y_test, axis=-1)
    print(f"X_train.shape : {X_train.shape}, Y_train.shape : {Y_train.shape}")  

    # concatenate X and Y (input dimention of LSTM will be 2)
    XY_train = np.concatenate([X_train, Y_train], axis=2)
    XY_test  = np.concatenate([X_test,  Y_test],  axis=2)
    XY_train = np.nan_to_num(XY_train, nan=0)
    mask = XY_train[:,-1,0] != 0
    print(mask)
    filtered = XY_train[mask]
    print(XY_train.shape)
    print(filtered.shape)
    print(XY_train)
    filtered = np.nan_to_num(filtered, nan=0)
    print(filtered)
    # plot_X(df_xtrain['data_missing'], df_xtrain['data_missing'], df_xtrain['data'], "test.png")
    # data_list = [df_xtrain['data_missing'], df_xtrain['data_missing'][3:].reset_index(drop=True)]
    # data_label = ['x train', 'y train']
    # plot_and_save(data_list, data_label, "test.png")

def main():
    # CSVファイルのパスを入力として受け取る
    file_path = os.path.expanduser("~/Sensor-Glove/src/data_handler/data/ind_0809_1951.csv")

    # 関数を呼び出してデータフレームを取得
    df = get_data(file_path)


    # add nan filter 
    df['rs4_filtered'] = df['rs4'].apply(lambda x: x if -90 <= x <= 150 else np.nan)
    df['sg1_filtered'] = df['sg1'].apply(lambda x: x if 100 <= x <= 600 else np.nan)


    # lowpass filter
    df['rs4_lowpass'] = fft_lowpass(df['rs4_filtered'], 5)
    # # plt.plot(df['sg1'])

    # scale data
    std_scaler = StandardScaler()
    std_scaler.fit(df)
    df_std = pd.DataFrame(std_scaler.transform(df), columns=df.columns)
    df_inverse = pd.DataFrame(std_scaler.inverse_transform(df_std), columns=df.columns)
    print(df_std.describe())
    # scaler_rs4 = StandardScaler()
    # scaler_rs4 = scaler_rs4.fit(df['rs4'].values.reshape(len(df['rs4']), 1))
    # df['rs4_filtered_scaled'] = scaler_rs4.transform(df['rs4_filtered'].values.reshape(len(df['rs4_filtered']), 1))
    # df['rs4_filtered_reverse'] = scaler_rs4.inverse_transform(df['rs4_filtered_scaled'].values.reshape(len(df['rs4_filtered_scaled']), 1))
    
    
    scaler_sg1 = StandardScaler()
    scaler_sg1 = scaler_sg1.fit(df['sg1'].values.reshape(len(df['sg1']), 1))
    df['sg1_filtered_scaled'] = scaler_sg1.transform(df['sg1_filtered'].values.reshape(len(df['sg1_filtered']), 1))
    df['sg1_filtered_reverse'] = scaler_sg1.inverse_transform(df['sg1_filtered_scaled'].values.reshape(len(df['sg1_filtered_scaled']), 1))

    # plot
    fig = plt.figure(figsize=(25, 5))
    # fig2 = plt.figure(figsize=(25, 5))
    ax = fig.add_subplot(211)
    ax2= fig.add_subplot(212)
    # ax.plot(df['rs4'], label='rs4')
    # ax.plot(df['rs5'], label='rs5')
    # ax.plot(df['sg1_filtered'], label='sg1_filtered')
    # ax.plot(df_inverse['sg1_filtered'], label='sg1_filtered_inverse')
    ax.plot(df['rs4_lowpass'], label='rs4_lowpass')
    ax2.plot(df['rs4_filtered'], label='rs4_std_filtered')
    # ax2.plot(df['rs4_filtered_scaled'], label='rs4_scaled')
    # ax2.plot(df['sg1_filtered_scaled'], label='sg1_scaled')
    # ax2.plot(df_std['sg1_filtered'], label='sg1_std_filtered')
    # ax2.plot(df['rs4_reverse'], label='rs4_reverse')
    # ax.plot(df['sg1_filtered'], label='sg1_filtered')
    # ax.plot(df['rs4_filtered'], label='rs4_filtered')

    
    
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    # main_dummy()
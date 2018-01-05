import numpy as np
import dataset.lh_build as build
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import my_lstm as MyModel

def result_LSTM():
    ts = 15
    dim = 1
    src = "dataset/matrix_no_zero.npy"
    src_hs300 = "dataset/hs300.npy"
    data_cache = "cache"
    train_split = 0.8
    val_split = 0.9

    x = Input(shape=(ts, dim))
    lstm = LSTM(32, input_shape=(ts, dim), return_sequences=True)(x)
    # lstm = LSTM(32, return_sequences=True)(lstm)
    prediction = Dense(1)(lstm)
    model = Model(input=x, output=[prediction, lstm])
    model.load_weights("encoder_snapshot.hdf5")
    X_target, Y_target, X_pos, Y_pos, X_neg, Y_neg, max_data, min_data, X_hs300, Y_hs300 = build.build_data(src,
                                                                                                            src_hs300,
                                                                                                            data_cache,
                                                                                                            target=9,
                                                                                                            timesteps=15,
                                                                                                            k=10,
                                                                                                            related_ts=15)
    train_index = round(train_split * X_target.shape[0])
    val_index = round(val_split * X_target.shape[0])

    X_train = X_target[:train_index, :, :]
    Y_train = Y_target[:train_index, :, :]
    print(Y_train.shape)
    predicted, encode = model.predict(X_train)
    Y_train = Y_train[:, -1, 0].flatten()
    predicted = predicted[:, -1, 0].flatten()
    train_error = np.sum((Y_train - predicted) ** 2) / Y_train.shape[0]
    print(train_error)

    X_val = X_target[train_index: val_index, :, :]
    Y_val = Y_target[train_index: val_index, :, :]
    print(Y_val.shape)
    predicted, encode = model.predict(X_val)
    Y_val = Y_val[:, -1, 0].flatten()
    predicted = predicted[:, -1, 0].flatten()
    val_error = np.sum((Y_val - predicted) ** 2) / Y_val.shape[0]
    print(val_error)
    delta_predict = predicted - X_val[:, -1, 0].flatten()
    delta_real = Y_val - X_val[:, -1, 0].flatten()
    p3 = delta_predict * delta_real

    count = 0
    for x in p3:
        if x > 0:
            count += 1
    accuracy = count / p3.shape[0]
    print(accuracy)

def exp_myModel():
    ts = 15
    dim = 1
    src = "dataset/matrix_no_zero.npy"
    src_hs300 = "dataset/hs300.npy"
    data_cache = "cache"
    train_split = 0.8
    val_split = 0.9
    x = Input(shape=(ts, dim))
    lstm = LSTM(32, input_shape=(ts, dim), return_sequences=True)(x)
    # lstm = LSTM(32, return_sequences=True)(lstm)
    prediction = Dense(1)(lstm)
    model = Model(input=x, output=[prediction, lstm])
    model.load_weights("encoder_snapshot.hdf5")
    X_target, Y_target, X_pos, Y_pos, X_neg, Y_neg, max_data, min_data, X_hs300, Y_hs300 = build.build_data(src,
                                                                                                            src_hs300,
                                                                                                            data_cache,
                                                                                                            target=0,
                                                                                                            timesteps=15,
                                                                                                            k=10,
                                                                                                            related_ts=15)

    X_pos = np.transpose(X_pos, (0, 2, 1))
    X_neg = np.transpose(X_neg, (0, 2, 1))
    X_ori = X_target
    [p, X_target] = model.predict(X_target)
    [p, X_hs300] = model.predict(X_hs300)
    X_pos_avg = np.zeros_like(X_target)
    X_neg_avg = np.zeros_like(X_target)

    for i in range(10):
        [p, temp] = model.predict(X_pos[:, :, i: i + 1])
        X_pos_avg += temp
        [p, temp] = model.predict(X_neg[:, :, i: i + 1])
        X_neg_avg += temp
    X_pos_avg /= 10
    X_neg_avg /= 10

    print(X_target.shape)
    print(X_pos_avg.shape)
    print(X_neg_avg.shape)
    print(X_hs300.shape)
    print(Y_target.shape)

    train_index = round(train_split * X_target.shape[0])
    val_index = round(val_split * X_target.shape[0])

    X_all = np.concatenate((X_target, X_pos_avg, X_neg_avg, X_hs300), axis=2)

    X_train_ori = X_ori[:train_index, :, :]
    X_train = X_all[:train_index, :, :]
    Y_train = Y_target[:train_index, :, :]
    print(X_train.shape)
    print(Y_train.shape)
    X_val_ori = X_ori[train_index: val_index, :, :]
    X_val = X_all[train_index: val_index, :, :]
    Y_val = Y_target[train_index: val_index, :, :]

    myModel = MyModel.myModel()
    epochs = 5000
    snapshot = 10
    best_val = np.inf
    accuracy_in_best_val = 0.0
    best_epoch = 0
    for i in range(int(epochs / snapshot)):
        myModel.fit(
            X_train,
            Y_train,
            batch_size=128,
            epochs=1)
        predicted = myModel.predict(X_train)
        train_error = np.sum((predicted[:, -1, 0] - Y_train[:, -1, 0]) ** 2) / predicted.shape[0]
        print("train error: %f" % train_error)

        predicted = myModel.predict(X_val)
        val_error = np.sum((predicted[:, -1, 0] - Y_val[:, -1, 0]) ** 2) / predicted.shape[0]
        print("val error: %f" % val_error)

        y_predict = np.array(predicted).flatten()
        y_real = np.array(Y_val).flatten()
        x_real = np.array(X_val_ori).flatten()
        delta_predict = y_predict - x_real
        delta_real = y_real - x_real
        p3 = delta_predict * delta_real
        count = 0
        for x in p3:
            if x > 0:
                count += 1
        accuracy = count / p3.shape[0]
        print("val accuracy: %f" % accuracy)

        if val_error < best_val:
            best_val = val_error
            accuracy_in_best_val = accuracy
            best_epoch = (i + 1) * snapshot
    print("best val error: %f" % best_val)
    print("accuracy : %.2f%%" % (accuracy_in_best_val * 100))
    print("best epoch: %d" % best_epoch)

if __name__ == "__main__":
    result_LSTM()
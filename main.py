import numpy as np
import dataset.lh_build as build
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *

def exp_LSTM(niter, nsnapshot, ts, x_train, y_train, x_val, y_val, dim=1):
    x = Input(shape=(ts, dim))
    lstm = LSTM(32, input_shape=(ts, dim), return_sequences=True)(x)
    prediction = Dense(1)(lstm)
    model = Model(input=x, output=prediction)
    model.compile('rmsprop', 'mse')

    best_error = np.inf
    best_accuracy = 0.0

    # args.nsnapshot denotes how many epochs per weight saving.
    for ii in range(int(niter / nsnapshot)):
        model.fit(
            x_train,
            y_train,
            batch_size=128,
            epochs=nsnapshot)

        num_iter = nsnapshot * (ii + 1)
        model.save_weights('snapshots/lstm/weights_ts%d_iter_%d.hdf5' % (ts, num_iter),
                           overwrite=True)

        predicted = model.predict(x_train)
        train_error = np.sum((predicted[:, -1, 0] - y_train[:, -1, 0]) ** 2) / predicted.shape[0]

        print('%s training error %f' % (num_iter, train_error))

        predicted = model.predict(x_val)
        val_error = np.sum((predicted[:, -1, 0] - y_val[:, -1, 0]) ** 2) / predicted.shape[0]

        p1 = np.array(predicted).flatten()
        p2 = np.array(y_val).flatten()
        p3 = p1 * p2
        count = 0
        for x in p3:
            if x > 0:
                count += 1
        accuracy = count / p3.shape[0]

        print('val error %f' % val_error)
        print('accuracy %.4f' % accuracy)

        if (accuracy > best_accuracy):
            best_error = val_error
            best_accuracy = accuracy
            best_iter = nsnapshot * (ii + 1)
        np.savetxt('results/lstm/best_iter_ts_%d.txt' % ts, [best_iter, best_error], fmt='%.6f')
        f = open('results/lstm/val_error_ts_%d.txt' % ts, 'a')
        f.write('iter=%d, val_error=%.6f, accuracy=%.4f\n' % (num_iter, val_error, accuracy))

    print('best iteration %d' % best_iter)
    print('smallest error %f' % best_error)
    print('best accuracy %.4f' % best_accuracy)

if __name__ == "__main__":
    src = "dataset/matrix_no_zero.npy"
    data_cache = "cache"
    train_split = 0.8
    val_split = 0.9

    X_target, Y_target, X_pos, Y_pos, X_neg, Y_neg, max_data, min_data = build.build_data(src, data_cache, 0, 15, 20, 30)

    X_pos = np.transpose(X_pos, (0, 2, 1))
    X_neg = np.transpose(X_neg, (0, 2, 1))
    Y_pos = np.transpose(Y_pos, (0, 2, 1))
    Y_neg = np.transpose(Y_neg, (0, 2, 1))

    train_border = round(train_split * X_target.shape[0])
    val_border = round(val_split * X_target.shape[0])

    X_train = X_target[: train_border, :, :]
    Y_train = Y_target[: train_border, :]

    for i in range(X_pos.shape[2]):
        X_train = np.concatenate((X_train, X_pos[: train_border, :, i: i+1]), axis=0)
        X_train = np.concatenate((X_train, X_neg[: train_border, :, i: i+1]), axis=0)
        Y_train = np.concatenate((Y_train, Y_pos[: train_border, :, i: i+1]), axis=0)
        Y_train = np.concatenate((Y_train, Y_neg[: train_border, :, i: i+1]), axis=0)

    X_val = X_target[train_border: val_border, :, :]
    Y_val = Y_target[train_border: val_border, :]

    exp_LSTM(100, 5, 15, X_train, Y_train, X_val, Y_val, dim=X_train.shape[2])


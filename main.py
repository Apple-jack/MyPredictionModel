import numpy as np
import dataset.lh_build as build
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from my_lstm import My_LSTM

def exp_LSTM(niter, nsnapshot, ts, x_train, y_train, x_val, y_val, dim=1, exp_try = 0, best_error = np.inf):
    x = Input(shape=(ts, dim))
    lstm = LSTM(32, input_shape=(ts, dim), return_sequences=True)(x)
    # lstm = LSTM(32, return_sequences=True)(lstm)
    prediction = Dense(1)(lstm)
    model = Model(input=x, output=prediction)
    model.compile('rmsprop', 'mse')

    best_accuracy = 0.0

    # args.nsnapshot denotes how many epochs per weight saving.
    for ii in range(int(niter / nsnapshot)):
        model.fit(
            x_train,
            y_train,
            batch_size=128,
            epochs=nsnapshot)

        num_iter = nsnapshot * (ii + 1)

        predicted = model.predict(x_train)
        train_error = np.sum((predicted[:, -1, 0] - y_train[:, -1, 0]) ** 2) / predicted.shape[0]

        print('%s train error %f' % (num_iter, train_error))
        print("train sample: %d" % predicted.shape[0])

        predicted = model.predict(x_val)
        val_error = np.sum((predicted[:, -1, 0] - y_val[:, -1, 0]) ** 2) / predicted.shape[0]

        y_predict = np.array(predicted).flatten()
        y_real= np.array(y_val).flatten()
        x_real = np.array(x_val).flatten()
        delta_predict = y_predict - x_real
        delta_real = y_real - x_real
        p3 = delta_predict * delta_real
        count = 0
        for x in p3:
            if x > 0:
                count += 1
        accuracy = count / p3.shape[0]

        print('val error %f' % val_error)
        print('accuracy %.4f' % accuracy)
        print("val sample: %d" % predicted.shape[0])

        if (val_error < best_error):
            best_error = val_error
            best_accuracy = accuracy
            best_iter = nsnapshot * (ii + 1)
            model.save_weights('snapshots/lstm/weights_ts%d_iter%d_try%d.hdf5' % (ts, num_iter, exp_try),
                               overwrite=True)
        np.savetxt('results/lstm/best_error_ts%d.txt' % ts, [best_iter, best_error, exp_try], fmt='%.6f')
        f = open('results/lstm/val_error_ts%d.txt' % ts, 'a')
        f.write('try=%d, iter=%d, val_error=%.6f, accuracy=%.4f\n' % (exp_try, num_iter, val_error, accuracy))

    print('best iteration %d' % best_iter)
    print('smallest error %f' % best_error)
    print('best accuracy %.4f' % best_accuracy)
    return best_error

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

    src = "dataset/matrix_no_zero.npy"
    # src = "dataset/data.npy"
    src_hs300 = "dataset/hs300.npy"
    data_cache = "cache"
    train_split = 0.8
    val_split = 0.9
    trys = 50

    # X_target, Y_target, X_pos, Y_pos, X_neg, Y_neg, max_data, min_data, X_hs300, Y_hs300 = build.build_data(src, src_hs300, data_cache, target=0, timesteps=15, k=20, related_ts=30)
    x_train, y_train, x_val, y_val, x_test, y_test, gt_test, max_data, min_data = build.load_data_2(src, src_hs300, 15, True)
    # # cache data
    # for i in range(20):
    #     print("start loading target: %d" % i)
    #     build.build_data(src, data_cache, target=i, timesteps=15, k=10, related_ts=30)

    # X_pos = np.transpose(X_pos, (0, 2, 1))
    # X_neg = np.transpose(X_neg, (0, 2, 1))
    # Y_pos = np.transpose(Y_pos, (0, 2, 1))
    # Y_neg = np.transpose(Y_neg, (0, 2, 1))
    #
    # train_border = round(train_split * X_target.shape[0])
    # val_border = round(val_split * X_target.shape[0])
    #
    # X_train = X_target[: train_border, :, :]
    # Y_train = Y_target[: train_border, :]
    #
    # for i in range(X_pos.shape[2]):
    #     X_train = np.concatenate((X_train, X_pos[: train_border, :, i: i+1]), axis=0)
    #     X_train = np.concatenate((X_train, X_neg[: train_border, :, i: i+1]), axis=0)
    #     Y_train = np.concatenate((Y_train, Y_pos[: train_border, :, i: i+1]), axis=0)
    #     Y_train = np.concatenate((Y_train, Y_neg[: train_border, :, i: i+1]), axis=0)
    #
    # X_val = X_target[train_border: val_border, :, :]
    # Y_val = Y_target[train_border: val_border, :]
    #
    best_error = np.inf
    for i in range(trys):
        best_error = exp_LSTM(niter=50, nsnapshot=1, ts=15, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                             dim=x_train.shape[2], exp_try=i, best_error=best_error)


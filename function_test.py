import numpy as np
import dataset.lh_build as build
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.layers import Average, Concatenate
from keras.models import *
from my_lstm import MulInput_LSTM
from lh_model import Proposed_Model

if __name__ == "__main__":
    ts = 15
    src = "dataset/matrix_no_zero.npy"
    src_hs300 = "dataset/hs300.npy"
    data_cache = "cache"
    X_target, Y_target, X_pos, Y_pos, X_neg, Y_neg, max_data, min_data, X_hs300, Y_hs300 = build.build_data(src,
                                                                                                            src_hs300,
                                                                                                            data_cache,
                                                                                                            target=0,
                                                                                                            timesteps=15,
                                                                                                            k=10,
                                                                                                            related_ts=15)
    k = 10
    Y_target = Y_target[:, -1:, 0]
    X_pos = X_pos.transpose(0, 2, 1)
    X_neg = X_neg.transpose(0, 2, 1)
    all_input_list = list()
    all_input_list_val = list()
    all_input_list.append(X_target[:750, :, :])
    all_input_list_val.append((X_target[750:, :, :]))
    for i in range(k):
        all_input_list.append(X_pos[:750, :, i: i + 1])
        all_input_list_val.append(X_pos[750:, :, i: i + 1])
    for i in range(k):
        all_input_list.append(X_neg[:750, :, i: i + 1])
        all_input_list_val.append(X_neg[750:, :, i: i + 1])
    all_input_list.append(X_hs300[:750, :, :])
    all_input_list_val.append(X_hs300[750:, :, :])

    model = Proposed_Model(k)


    best_accuracy = 0.0
    best_iter = 0
    niter = 1000
    nsnapshot = 5
    best_error = np.inf
    # args.nsnapshot denotes how many epochs per weight saving.
    for ii in range(int(niter / nsnapshot)):
        model.fit(all_input_list, [Y_target[:750, :], Y_target[:750, :]], batch_size=128, epochs=nsnapshot)


        num_iter = nsnapshot * (ii + 1)

        [ap, predicted] = model.predict(all_input_list)
        train_error = np.sum((predicted.flatten() - Y_target[:750, :].flatten()) ** 2) / predicted.shape[0]
        print('%s train error %f' % (num_iter, train_error))
        print("train sample: %d" % predicted.shape[0])

        [ap, predicted] = model.predict(all_input_list_val)
        val_error = np.sum((predicted.flatten() - Y_target[750:, :].flatten()) ** 2) / predicted.shape[0]

        y_predict = np.array(predicted).flatten()
        y_real = np.array(Y_target[750:, :]).flatten()
        x_real = np.array(X_target[750:, -1, 0]).flatten()

        delta_predict = y_predict - x_real
        delta_real = y_real - x_real
        p3 = delta_predict * delta_real
        # print(delta_predict)
        # print("--------------------------------------------------------")
        # print(delta_real)
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

    print('best iteration %d' % best_iter)
    print('smallest error %f' % best_error)
    print('best accuracy %.4f' % best_accuracy)
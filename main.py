import numpy as np
import dataset.lh_build as build
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from lh_model import Proposed_Model, AttentionLayer

def evalueate(model, type, x, y, input_list=0):
    # if type == 'LSTM':
    #     predicted = model.predict(x).flatten()
    #     error = np.sum((predicted - y.flatten()) ** 2) / predicted.shape[0]
    #     ## calculate accuracy
    #     y_real = np.array(y).flatten()
    #     x_real = x[:, -1, 0].flatten()
    #     delta_predict = predicted - x_real
    #     delta_real = y_real - x_real
    #     flag = delta_predict * delta_real
    #     count = 0
    #     for _ in flag:
    #         if _ > 0:
    #             count += 1
    #     accuracy = count / flag.shape[0]
    #     return error, accuracy, predicted.shape[0]
    # elif type == 'proposed_model':
    if type == 'LSTM' or type == 'proposed':
        if type == 'LSTM':
            aux_predicted, predicted, tag_predicted = model.predict(input_list)
        else:
            predicted, aux_predicted, tag_predicted = model.predict(input_list)
        aux_predicted = np.array(aux_predicted).flatten()
        predicted = np.array(predicted).flatten()
        error = np.sum((predicted - y.flatten()) ** 2) / predicted.shape[0]

        aux_error = np.sum((aux_predicted - y.flatten()) ** 2) / aux_predicted.shape[0]

        ## calculate accuracy
        y_real = np.array(y).flatten()
        x_real = np.array(x)[:, -1, 0].flatten()
        delta_predict = predicted - x_real
        delta_aux_predict = aux_predicted - x_real
        delta_real = y_real - x_real
        flag = delta_predict * delta_real
        aux_flag = delta_aux_predict * delta_real
        count = 0
        aux_count = 0
        for sample in flag:
            if sample > 0:
                count += 1
        for sample in aux_flag:
            if sample > 0:
                aux_count += 1
        accuracy = count / flag.shape[0]
        aux_accuracy = aux_count / aux_flag.shape[0]
        return error, accuracy, aux_error, aux_accuracy, predicted.shape[0]

    else:
        print('unknow experiment type')
        return 0

def exp_LSTM(niter, nsnapshot, ts, train, val, test, dim=22, exp_try = 0, best_error = np.inf, best_accuracy = 0.0, return_sequences=True, targets=[0]):
    ## 2 layer LSTM
    ## notice that y_train has 3 dims: (normalized_value, max_raw_value, min_raw_value)
    ## we should only use the first dimension for training
    x = Input(shape=(ts, dim))
    lstm = LSTM(64, input_shape=(ts, dim), return_sequences=True)(x)
    lstm = LSTM(64, return_sequences=True)(lstm)
    lstm = AttentionLayer(output_dim=64*2, timesteps=ts)(lstm)
    prediction = Dense(1)(lstm)
    model = Model(input=x, output=prediction)
    model.compile('rmsprop', 'mse')
    # x_train = np.array(train[0])
    x_train = np.concatenate((train[0], train[3], train[4], train[5]), axis=2)
    y_train = np.array(train[1][:, 0: 1]      )  ## use only the first dimension
    # x_val = np.array(val[0])
    x_val = np.concatenate((val[0], val[3], val[4], val[5]), axis=2)
    y_val = np.array(val[1][:, 0: 1])            ## use only the first dimension
    # x_test = np.array(test[0])
    x_test = np.concatenate((test[0], test[3], test[4], test[5]), axis=2)
    y_test = np.array(test[1][:, 0: 1])          ## use only the first dimension
    ## results under best mse
    best_mse_iter = 0
    best_mse_acc = 0.0
    best_mse_test_mse = 0.0
    best_mse_test_acc = 0.0
    ## results under best accuracy
    best_acc_iter = 0
    best_acc_mse = 0.0
    best_acc_test_mse = 0.0
    best_acc_test_acc =0.0

    # args.nsnapshot denotes how many epochs per weight saving.
    for ii in range(int(niter / nsnapshot)):
        model.fit(
            x_train,
            y_train,
            batch_size=128,
            epochs=nsnapshot)

        num_iter = nsnapshot * (ii + 1)
        ########################## train error
        train_error, train_accuracy, n_sample = evalueate(model, 'LSTM', x_train, y_train)
        print('%s train error %f' % (num_iter, train_error))
        print('accuracy %.4f' % train_accuracy)
        print("train sample: %d" % n_sample)
        ########################## validate error
        val_error, val_accuracy, n_sample = evalueate(model, 'LSTM', x_val, y_val)
        print('val error %f' % val_error)
        print('accuracy %.4f' % val_accuracy)
        print("val sample: %d" % n_sample)
        ########################## test error
        test_error, test_accuracy, n_sample = evalueate(model, 'LSTM', x_test, y_test)
        print('test error %f' % test_error)
        print('accuracy %.4f' % test_accuracy)
        print("test sample: %d" % n_sample)

        if (val_error < best_error):
            best_error = val_error
            best_mse_iter = num_iter
            best_mse_acc = val_accuracy
            best_mse_test_mse = test_error
            best_mse_test_acc = test_accuracy
            model.save_weights('snapshots/lstm/weights_ts%d_iter%d_try%d_target%d.hdf5' % (ts, num_iter, exp_try, targets[0]),
                               overwrite=True)

        if (val_accuracy > best_accuracy):
            best_accuracy = val_accuracy
            best_acc_iter = num_iter
            best_acc_mse = val_error
            best_acc_test_mse = test_error
            best_acc_test_acc = test_accuracy
            model.save_weights('snapshots/lstm/weights_ts%d_iter%d_try%d_target%d.hdf5' % (ts, num_iter, exp_try, targets[0]),
                               overwrite=True)

        f = open('results/lstm/val_error_target%d.txt' % targets[0], 'a')
        f.write('try=%d, iter=%d, train_error=%.6f, val_error=%.6f, accuracy=%.4f, test_error=%.6f, accuracy=%.4f\n'
                % (exp_try, num_iter, train_error, val_error, val_accuracy, test_error, test_accuracy))
        f.close()
    ## train over
    f = open('results/lstm/best_error_target%d.csv' % targets[0], 'a')
    f.write('%d, %f, %d, %f, %f, %f,'
            ' %f, %d, %f, %f, %f\n'
            % (exp_try, best_error, best_mse_iter, best_mse_acc, best_mse_test_mse, best_mse_test_acc,
               best_accuracy, best_acc_iter, best_acc_mse, best_acc_test_mse, best_acc_test_acc))
    f.close()

    print('best mse iteration %d' % best_mse_iter)
    print('smallest error %f' % best_error)
    print('best accuracy iteration %d' % best_acc_iter)
    print('best accuracy %.4f' % best_accuracy)
    return best_error, best_accuracy

def exp_Proposed(niter, nsnapshot, ts, train, val, test, lstm_dim, exp_try = 0, best_error = np.inf, best_accuracy = 0.0, type='proposed', targets=[0]):
    ## 2 layer LSTM
    ## notice that y_train has 3 dims: (normalized_value, max_raw_value, min_raw_value)
    ## we should only use the first dimension for training
    k = 10
    model = Proposed_Model(k, ts=ts, lstm_dim=64, type=type)

    x_train = np.array(train[0])
    y_train = np.array(train[1][:, 0: 1])  ## use only the first dimension
    x_val = np.array(val[0])
    y_val = np.array(val[1][:, 0: 1])  ## use only the first dimension
    x_test = np.array(test[0])
    y_test = np.array(test[1][:, 0: 1])  ## use only the first dimension

    all_input_list = list()
    all_input_list_val = list()
    all_input_list_test = list()
    all_input_list.append(train[0])
    all_input_list_val.append(val[0])
    all_input_list_test.append(test[0])
    for i in range(k):
        all_input_list.append(train[3][:, :, i: i + 1])
        all_input_list_val.append(val[3][:, :, i: i + 1])
        all_input_list_test.append(test[3][:, :, i: i + 1])
    for i in range(k):
        all_input_list.append(train[4][:, :, i: i + 1])
        all_input_list_val.append(val[4][:, :, i: i + 1])
        all_input_list_test.append(test[4][:, :, i: i + 1])
    all_input_list.append(train[5])
    all_input_list_val.append(val[5])
    all_input_list_test.append(test[5])

    tag_train = np.array(train[2])
    tag_val = np.array(val[2])

    ## results under best mse
    best_mse_iter = 0
    best_mse_acc = 0.0
    best_mse_aux_mse = 0.0
    best_mse_aux_acc = 0.0
    best_mse_test_mse = 0.0
    best_mse_test_acc = 0.0
    best_mse_test_aux_mse = 0.0
    best_mse_test_aux_acc = 0.0
    ## results under best accuracy
    best_acc_iter = 0
    best_acc_mse = 0.0
    best_acc_aux_mse = 0.0
    best_acc_aux_acc = 0.0
    best_acc_test_mse = 0.0
    best_acc_test_acc = 0.0
    best_acc_test_aux_mse = 0.0
    best_acc_test_aux_acc = 0.0

    # args.nsnapshot denotes how many epochs per weight saving.
    for ii in range(int(niter / nsnapshot)):
        model.fit(all_input_list, [y_train, y_train, tag_train], batch_size=512, epochs=nsnapshot, shuffle=True)

        num_iter = nsnapshot * (ii + 1)

        ## train error
        train_error, train_accuracy, train_aux_error, train_aux_accuracy, n_sample = evalueate(model, type, x_train, y_train, input_list=all_input_list)
        print('%s train error %f' % (num_iter, train_error))
        print('accuracy %.4f' % train_accuracy)
        print("train sample: %d" % n_sample)
        ## validate error
        val_error, val_accuracy, val_aux_error, val_aux_accuracy, n_sample = evalueate(model, type, x_val, y_val, input_list=all_input_list_val)
        print('val error %f' % val_error)
        print('accuracy %.4f' % val_accuracy)
        print("val sample: %d" % n_sample)
        ## test error
        test_error, test_accuracy, test_aux_error, test_aux_accuracy, n_sample = evalueate(model, type, x_test, y_test, input_list=all_input_list_test)
        print('test error %f' % test_error)
        print('accuracy %.4f' % test_accuracy)
        print("test sample: %d" % n_sample)

        if (val_error < best_error):
            best_error = val_error
            best_mse_iter = num_iter
            best_mse_acc = val_accuracy
            best_mse_aux_mse = val_aux_error
            best_mse_aux_acc = val_aux_accuracy
            best_mse_test_mse = test_error
            best_mse_test_acc = test_accuracy
            best_mse_test_aux_mse = test_aux_error
            best_mse_test_aux_acc = test_aux_accuracy
            model.save_weights(
                'snapshots/proposed/weights_ts%d_iter%d_try%d_target%d.hdf5' % (ts, num_iter, exp_try, targets[0]),
                overwrite=True)

        if (val_accuracy > best_accuracy):
            best_accuracy = val_accuracy
            best_acc_iter = num_iter
            best_acc_mse = val_error
            best_acc_aux_mse = val_aux_error
            best_acc_aux_acc = val_aux_accuracy
            best_acc_test_mse = test_error
            best_acc_test_acc = test_accuracy
            best_acc_test_aux_mse = test_aux_error
            best_acc_test_aux_acc = test_aux_accuracy
            model.save_weights(
                'snapshots/proposed/weights_ts%d_iter%d_try%d_target%d.hdf5' % (ts, num_iter, exp_try, targets[0]),
                overwrite=True)

        f = open('results/proposed/val_error_target%d.txt' % targets[0], 'a')
        f.write('try=%d, iter=%d, train_error=%.6f, val_error=%.6f, accuracy=%.4f,'
                'aux_error=%.6f, aux_accuracy=%.4f, test_error=%.6f, accuracy=%.4f,'
                'aux_error=%.6f, aux_accuracy=%.4f\n'
                % (exp_try, num_iter, train_error, val_error, val_accuracy,
                   val_aux_error, val_aux_accuracy, test_error, test_accuracy,
                   test_aux_error, test_aux_accuracy))
        f.close()
    ## train over
    f = open('results/proposed/best_error_target%d.csv' % targets[0], 'a')
    f.write('%d, %f, %d, %f, %f, %f, '
            '%f, %f, %f, %f,'
            ' %f, %d, %f, %f, %f, '
            '%f, %f, %f, %f\n'
            % (exp_try, best_error, best_mse_iter, best_mse_acc, best_mse_aux_mse, best_mse_aux_acc,
               best_mse_test_mse, best_mse_test_acc, best_mse_test_aux_mse, best_mse_test_aux_acc,
               best_accuracy, best_acc_iter, best_acc_mse, best_acc_aux_mse, best_acc_aux_acc,
               best_acc_test_mse, best_acc_test_acc, best_acc_test_aux_mse, best_acc_test_aux_acc))
    f.close()

    print('best mse iteration %d' % best_mse_iter)
    print('smallest error %f' % best_error)
    print('best accuracy iteration %d' % best_acc_iter)
    print('best accuracy %.4f' % best_accuracy)
    return best_error, best_accuracy

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

    src = "dataset/matrix.npy"
    src_hs300 = "dataset/hs300.npy"
    data_cache = "cache"
    trys = 1
    # best_error = np.inf
    # best_accuracy = 0.0
    #
    cailiao = [3, 22, 26, 35, 41, 43, 74, 76, 78, 80, 88, 101, 111, 116, 135, 145, 151, 154, 155, 167, 173, 174, 177, 219, 229, 251, 256, 260]
    # targets = [0]
    ## LSTM experiment
    # for t in range(261):
    #     print('training: target=%d' % t)
    #     train, val, test = build.get_train_val_test(type='proposed_model', src=src, src_hs300=src_hs300, data_cache=data_cache,
    #                                                 targets=[t])
    #     f = open('results/lstm/best_error_target%d.csv' % t, 'w')
    #     f.write('Experiment, Best Validate MSE, Iter, Accuracy, Test MSE, Test Accuracy,'
    #             'Best Validate Accuracy, Iter, MSE, Test MSE, Test Accuracy\n')
    #     f.close()
    #     for i in range(trys):
    #         best_error, best_accuracy = exp_LSTM(niter=1500, nsnapshot=5, ts=10, train=train, val=val, test=test,
    #                              dim=22, exp_try=i, return_sequences=False, targets=[t])

    ## Proposed Model experiment
    # for t in range(1):
    #     print('training: target=%d' % t)
    train, val, test = build.get_train_val_test(type='proposed_model', src=src, src_hs300=src_hs300,
                                                data_cache=data_cache, targets=cailiao, k=6)
    # f = open('results/proposed/best_error_target%s.csv' % 'cailiao', 'w')
    # f.write('Experiment, Best Validate MSE, Iter, Accuracy, Aux-MSE, Aux-Accuracy, '
    #         'Test MSE, Test Accuracy, Aux-MSE, Aux-Accuracy,'
    #         'Best Validate Accuracy, Iter, MSE, Aux-MSE, Aux-Accuracy, '
    #         'Test MSE, Test Accuracy, Aux-MSE, Aux-Accuracy\n')
    # f.close()
    # for i in range(trys):
    #     best_error, best_accuracy = exp_Proposed(niter=1500, nsnapshot=5, ts=10, train=train, val=val, test=test,
    #                                              lstm_dim=64, exp_try=i, type='LSTM', targets=cailiao)

    ## Evaluate
    # train, val, test = build.get_train_val_test(type='proposed_model', src=src, src_hs300=src_hs300,
    #                                             data_cache=data_cache, targets=[0])
    # cal_test_error(val, 'proposed_model', 15, 0, 1960)
    # cal_test_error(test, 'proposed_model', 15, 0, 1525)

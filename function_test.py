import numpy as np
import dataset.lh_build as build
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.layers import Average
from keras.models import *
from my_lstm import MulInput_LSTM
import pandas as pd

def encoder_lstm(ts = 15, hidden_dim=32):
    seq = Sequential()
    seq.add(LSTM(hidden_dim, input_shape=(ts, 1), return_sequences=True))
    return seq

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

    X_pos = X_pos.transpose(0, 2, 1)
    X_neg = X_neg.transpose(0, 2, 1)
    print(X_target.shape)
    print(Y_target.shape)
    print(X_pos.shape)
    print(X_hs300.shape)

    x_pos = list()


    x_target = Input(shape=(ts, 1))
    x_hs300 = Input(shape=(ts, 1))
    x_pos = list()
    x_pos_out = list()
    x_neg = list()
    x_neg_out = list()
    encoder = encoder_lstm()

    ## positive and negative inputs
    for i in range(k):
        in_temp = Input(shape=(ts, 1))
        x_pos.append(in_temp)                 ## shape=(batch, ts, 1)
        in_temp = Input(shape=(ts, 1))
        x_neg.append(in_temp)                 ## shape=(batch, ts, 1)
    ## target and hs300 index encode outputs
    x_target_out = encoder(x_target)          ## shape=(batch, ts, dim)
    x_hs300_out = encoder(x_target)           ## shape=(batch, ts, dim)
    ## positive and negative encode outputs
    for i in range(k):
        x_pos_out.append(encoder(x_pos[i]))   ## shape=(batch, ts, dim)
        x_neg_out.append(encoder(x_neg[i]))   ## shape=(batch, ts, dim)
    ## auxilary output for target prediction
    aux_out = LSTM(32, return_sequences=False)(x_target_out)
    aux_out = Dense(1)(aux_out)
    ## concatenate positive and nagative series
    ## here use a naive average function to concatenate, edit Average layer source code to implement attention
    x_pos_avg = Average()(x_pos_out)          ## shape=(batch, ts, dim)
    x_neg_avg = Average()(x_neg_out)          ## shape=(batch, ts, dim)
    ## use a Average layer to pretend MulInput_LSTM
    x_fake = Average()([x_target_out, x_pos_avg, x_neg_avg, x_hs300_out])
    main_out = LSTM(32, return_sequences=False)(x_fake)
    main_out = Dense(1)(main_out)

    all_input_list = list()
    all_input_list.append(x_target)
    all_input_list += x_pos
    all_input_list += x_neg
    all_input_list.append(x_hs300)
    model = Model(inputs=tuple(all_input_list), outputs=(aux_out, main_out))
    model.compile('rmsprop', 'mse')

    model.fit()

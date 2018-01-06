import numpy as np

from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers import Average, Concatenate
from my_lstm import MulInput_LSTM

import dataset.lh_build as build

def encoder_lstm(ts = 15, lstm_dim=32):
    seq = Sequential()
    seq.add(LSTM(lstm_dim, input_shape=(ts, 1), return_sequences=True))
    return seq

def Proposed_Model(k, ts=15, lstm_dim=32):
    x_target = Input(shape=(ts, 1))
    x_hs300 = Input(shape=(ts, 1))
    x_pos = list()
    x_pos_out = list()
    x_neg = list()
    x_neg_out = list()
    encoder = encoder_lstm(ts=ts, lstm_dim=lstm_dim)

    ## positive and negative inputs
    for i in range(k):
        in_temp = Input(shape=(ts, 1))
        x_pos.append(in_temp)  ## shape=(batch, ts, 1)
        in_temp = Input(shape=(ts, 1))
        x_neg.append(in_temp)  ## shape=(batch, ts, 1)
    ## target and hs300 index encode outputs
    x_target_out = encoder(x_target)  ## shape=(batch, ts, dim)
    x_hs300_out = encoder(x_target)  ## shape=(batch, ts, dim)
    ## positive and negative encode outputs
    for i in range(k):
        x_pos_out.append(encoder(x_pos[i]))  ## shape=(batch, ts, dim)
        x_neg_out.append(encoder(x_neg[i]))  ## shape=(batch, ts, dim)
    ## auxilary output for target prediction
    aux_out = LSTM(lstm_dim, return_sequences=False)(x_target_out)
    aux_out = Dense(1)(aux_out)
    ## concatenate positive and nagative series
    ## here use a naive average function to concatenate, edit Average layer source code to implement attention
    x_pos_avg = Average()(x_pos_out)  ## shape=(batch, ts, dim)
    x_neg_avg = Average()(x_neg_out)  ## shape=(batch, ts, dim)
    ## concatenate the inputs to fit MulInput_LSTM input shape
    mul_LSTM_in = Concatenate(axis=2)([x_target_out, x_pos_avg, x_neg_avg, x_hs300_out])
    main_out = MulInput_LSTM(lstm_dim, return_sequences=False)(mul_LSTM_in)
    main_out = Dense(1)(main_out)

    all_input_list = list()
    all_input_list.append(x_target)
    all_input_list += x_pos
    all_input_list += x_neg
    all_input_list.append(x_hs300)
    model = Model(inputs=tuple(all_input_list), outputs=(aux_out, main_out))
    model.compile('rmsprop', 'mse')
    return  model

if __name__ == "__main__":
    encode_dim = 1

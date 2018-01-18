import numpy as np

from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers import Average, Concatenate, LeakyReLU
from my_lstm import MulInput_LSTM

import dataset.lh_build as build

class AttentionLayer(Layer):

    def __init__(self, output_dim, timesteps, **kwargs):
        self.output_dim = output_dim
        self.timesteps = timesteps
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='attention_kernel',
                                      shape=(input_shape[2], input_shape[2]),
                                      initializer='uniform',
                                      trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        alpha = []
        for i in range(0, self.timesteps - 1):
            a = K.dot(x[:, i, :], self.kernel)
            a = a * x[:, -1, :]
            a = K.sum(a, axis=1, keepdims=True)
            alpha.append(a)
        alpha = activations.softmax(K.concatenate(tuple(alpha), axis=1))
        c = alpha[:, 0: 1] * x[:, 0, :]
        for i in range(1, self.timesteps - 1):
            c = c + alpha[:, i: i + 1] * x[:, i, :]
        h = K.concatenate((c, x[:, -1, :]), axis=1)
        return h

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def encoder_lstm(ts = 15, lstm_dim=64, trainable=False):
    seq = Sequential()
    seq.add(LSTM(lstm_dim, input_shape=(ts, 1), return_sequences=True, trainable=trainable))
    return seq

def Proposed_Model(k, ts=15, lstm_dim=64, type='LSTM'):
    if type == 'LSTM':
        lstm_flag = True
        proposed_flag = False
    else:
        lstm_flag = False
        proposed_flag = True

    x_target = Input(shape=(ts, 1))
    x_hs300 = Input(shape=(ts, 1))
    x_pos = list()
    x_pos_out = list()
    x_neg = list()
    x_neg_out = list()
    encoder = encoder_lstm(ts=ts, lstm_dim=lstm_dim, trainable=True)

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
    aux_out = LSTM(lstm_dim, return_sequences=True, trainable=lstm_flag)(x_target_out)
    aux_out = AttentionLayer(output_dim=lstm_dim*2, timesteps=ts, trainable=lstm_flag)(aux_out)
    aux_out = Dense(1, name='aux', activation=None, trainable=lstm_flag)(aux_out)

    ## concatenate positive and nagative series
    ## here use a naive average function to concatenate, edit Average layer source code to implement attention
    x_pos_avg = Average()(x_pos_out)  ## shape=(batch, ts, dim)
    x_neg_avg = Average()(x_neg_out)  ## shape=(batch, ts, dim)
    ## concatenate the inputs to fit MulInput_LSTM input shape
    mul_LSTM_in = Concatenate(axis=2)([x_target_out, x_pos_avg, x_neg_avg, x_hs300_out])
    mul_LSTM_out = MulInput_LSTM(lstm_dim, return_sequences=True, trainable=proposed_flag)(mul_LSTM_in)
    # mul_LSTM_out = LSTM(lstm_dim, return_sequences=True)(mul_LSTM_in)
    # mul_LSTM_out = LSTM(lstm_dim, return_sequences=True)(x_target_out)
    atn_out = AttentionLayer(output_dim=lstm_dim * 2, timesteps=ts, trainable=proposed_flag)(mul_LSTM_out)
    atn_out = Dense(32, activation='relu', trainable=proposed_flag)(atn_out)
    atn_out = Dense(32, activation='relu', trainable=proposed_flag)(atn_out)
    main_out = Dense(1, name='main', activation=None, trainable=proposed_flag)(atn_out)

    classify_out = Dense(1, activation='sigmoid', name='classify')(atn_out)

    all_input_list = list()
    all_input_list.append(x_target)
    all_input_list += x_pos
    all_input_list += x_neg
    all_input_list.append(x_hs300)
    model = Model(inputs=tuple(all_input_list), outputs=(main_out, aux_out, classify_out))
    if type == 'LSTM':
        w_aux = 1
        w_main = 0
    else:
        model.load_weights('weights_ts10_iter485_try0_target0.hdf5')
        print('weights loaded')
        w_aux = 0
        w_main = 1
    model.compile(optimizer='rmsprop',
                  loss={'aux': 'mse', 'main': 'mse', 'classify': 'binary_crossentropy'},
                  loss_weights={'aux': w_aux, 'main': w_main, 'classify': 0})
    return  model

if __name__ == "__main__":
    k = 10
    Proposed_Model(k, lstm_dim=128)
    # ts = 10
    # dim = 32
    # data_x = np.random.standard_normal(size=(5, ts, dim))
    # data_y = np.random.standard_normal(size=(5, 256))
    # x = Input(shape=(ts, dim))
    # lstm = LSTM(128, input_shape=(ts, dim), return_sequences=True)(x)
    # print(lstm.shape)
    # lstm = AttentionLayer(output_dim=256, timesteps=ts)(lstm)
    # print(lstm.shape)
    # model = Model(inputs=x, outputs=lstm)
    # model.compile('rmsprop', 'mse')
    # model.fit(data_x, data_y)

import numpy as np
import dataset.lh_build as build
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.layers import Average, Concatenate
from keras.models import *
from keras import backend as K
from keras.engine.topology import Layer
from my_lstm import MulInput_LSTM
from lh_model import Proposed_Model

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

if __name__ == "__main__":
    ts = 10
    dim = 32
    data_x = np.random.standard_normal(size=(5, ts, dim))
    x = Input(shape=(ts, dim))
    lstm = LSTM(128, input_shape=(ts, dim), return_sequences=True)(x)
    print(lstm.shape)
    lstm = AttentionLayer(output_dim=ts, timesteps=ts)(lstm)
    print(lstm.shape)
    model = Model(inputs=x, outputs=lstm)
    model.compile('rmsprop', 'mse')
    print(model.predict(data_x))
    # ts = 15
    # src = "dataset/matrix.npy"
    # src_hs300 = "dataset/hs300.npy"
    # data_cache = "cache"
    # X_target, Y_target, X_pos, Y_pos, X_neg, Y_neg, max_data, min_data, X_hs300, Y_hs300 = build.build_data(src,
    #                                                                                                         src_hs300,
    #                                                                                                         data_cache,
    #                                                                                                         target=0,
    #                                                                                                         timesteps=15,
    #                                                                                                         k=10,
    #                                                                                                         related_ts=30)
    # print(X_target.shape)
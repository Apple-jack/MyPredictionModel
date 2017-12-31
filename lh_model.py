import numpy as np

from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import ones
from keras.constraints import min_max_norm

import dataset.lh_build as build

class Filter(Layer):

    def __init__(self, encode_dim, output_dim, **kwargs):
        self.encode_dim = encode_dim
        self.output_dim = output_dim
        super(Filter, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, self.encode_dim),
                                      initializer=ones(),
                                      constraint=min_max_norm(min_value=0.0, max_value=1.0000000001, rate=1.0, axis=0),
                                      trainable=True)
        super(Filter, self).build(input_shape)

    def call(self, inputs):
        target = inputs[:, :self.encode_dim]
        driving = inputs[:, self.encode_dim:]

        target = target * self.kernel
        f = self.kernel
        driving = driving * (1 - f)
        o = K.concatenate((target, driving), axis=1)
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


if __name__ == "__main__":
    encode_dim = 1
    output_dim = 2

    x_in = np.random.standard_normal(size=(100, 2))
    y_out = np.random.standard_normal(size=(100, 1))
    y_out = x_in[:, 1]

    x = Input(shape=(encode_dim * 2,))
    fil = Filter(encode_dim=encode_dim, output_dim=output_dim)(x)
    out = Dense(1)(fil)
    model = Model(inputs=x, outputs=(out,fil))
    model.load_weights("wwwww.hdf5")
    print(model.predict(np.array([[1, 1]])))
    # model.compile('rmsprop', 'mse')
    # model.fit(x_in, y_out, epochs=5000)
    #
    # result = model.predict(x_in)
    # print(result - y_out)
    # model.save_weights("wwwww.hdf5")



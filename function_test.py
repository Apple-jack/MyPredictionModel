import numpy as np
import dataset.lh_build as build
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from my_lstm import MulInput_LSTM

if __name__ == "__main__":
    ts = 15
    dim = 40
    x = Input(shape=(ts, dim))
    lstm = MulInput_LSTM(32, input_shape=(ts, dim), return_sequences=True)(x)
    # lstm = LSTM(32, return_sequences=True)(lstm)
    prediction = Dense(1)(lstm)
    model = Model(input=x, output=prediction)
    model.compile('rmsprop', 'mse')

    testx = np.random.standard_normal(size=(10, 15, 40))
    testy = np.ones(shape=(10, 15, 1))

    # print(model.predict(testx))
    model.fit(testx, testy, epochs=100)
    # model.predict(testx)
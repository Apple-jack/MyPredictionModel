# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.engine import Layer, InputSpec
from keras.legacy import interfaces
from keras.layers.recurrent import Recurrent

def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):

    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x

class MulInput_LSTM(Recurrent):
    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(MulInput_LSTM, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))
        self.state_spec = [InputSpec(shape=(batch_size, self.units)),
                           InputSpec(shape=(batch_size, self.units))]

        self.states = [None, None]
        if self.stateful:
            self.reset_states()

        self.series_num = 4     ## we use 4 series: target, positive, negative, hs300
        ## kernels: kernel_f,
        ##          kernel_i, kernel_ip, kernel_in, kernel_ihs
        ##          kernel_c, kernel_cp, kernel_cn, kernel_chs
        ##          kernel_o

        ## recurrent kernels: recurrent_kernel_f,
        ##                    recurrent_kernel_i, recurrent_kernel_ip, recurrent_kernel_in, recurrent_kernel_ihs
        ##                    recurrent_kernel_c, recurrent_kernel_cp, recurrent_kernel_cn, recurrent_kernel_chs
        ##                    recurrent_kernel_o
        ## total: 20 kernels

        ## these are the original kernels for LSTM
        ## total: 8 kernels
        self.kernel = self.add_weight((int(self.input_dim / 4), self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            (self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        ## here we use another variable to specify driving series kernel, since they use different initializer
        ## kernel will be zero at the beginning
        ## total: 12 kernels
        # self.kernel_rel = self.add_weight((int(self.input_dim / self.series_num), self.units * 6),
        #                               name='kernel_related',
        #                               initializer=initializers.zeros(),
        #                               regularizer=self.kernel_regularizer,
        #                               constraint=self.kernel_constraint)
        #
        # self.recurrent_kernel_rel = self.add_weight(
        #     (self.units, self.units * 6),
        #     name='recurrent_kernel_related',
        #     initializer=initializers.zeros(),
        #     regularizer=self.recurrent_regularizer,
        #     constraint=self.recurrent_constraint)

        ## original 4 bias and 6 related bias
        ## all bias use the same initializer
        if self.use_bias:
            self.bias = self.add_weight((self.units * 4,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            if self.unit_forget_bias:
                bias_value = np.zeros((self.units * 4,))
                bias_value[self.units: self.units * 2] = 1.
                K.set_value(self.bias, bias_value)
        else:
            self.bias = None

        ## original kernels
        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        ## related kernels
        # self.kernel_ip = self.kernel_rel[:, :self.units]
        # self.kernel_in = self.kernel_rel[:, self.units: self.units * 2]
        # self.kernel_ihs = self.kernel_rel[:, self.units * 2: self.units * 3]
        # self.kernel_cp = self.kernel_rel[:, self.units * 3: self.units * 4]
        # self.kernel_cn = self.kernel_rel[:, self.units * 4: self.units * 5]
        # self.kernel_chs = self.kernel_rel[:, self.units * 5:]
        #
        # self.recurrent_kernel_ip = self.recurrent_kernel_rel[:, :self.units]
        # self.recurrent_kernel_in = self.recurrent_kernel_rel[:, self.units: self.units * 2]
        # self.recurrent_kernel_ihs = self.recurrent_kernel_rel[:, self.units * 2: self.units * 3]
        # self.recurrent_kernel_cp = self.recurrent_kernel_rel[:, self.units * 3: self.units * 4]
        # self.recurrent_kernel_cn = self.recurrent_kernel_rel[:, self.units * 4: self.units * 5]
        # self.recurrent_kernel_chs = self.recurrent_kernel_rel[:, self.units * 5:]

        if self.use_bias:
            ## original bias
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]

            ## related bias
            # self.bias_ip = self.bias[self.units * 4: self.units * 5]
            # self.bias_in = self.bias[self.units * 5: self.units * 6]
            # self.bias_ihs = self.bias[self.units * 6: self.units * 7]
            # self.bias_cp = self.bias[self.units * 7: self.units * 8]
            # self.bias_cn = self.bias[self.units * 8: self.units * 9]
            # self.bias_chs = self.bias[self.units * 9:]

        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
            # self.bias_ip = None
            # self.bias_in = None
            # self.bias_ihs = None
            # self.bias_cp = None
            # self.bias_cn = None
            # self.bias_chs = None
        self.built = True

    def get_constants(self, inputs, training=None):
        constants = []
        constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        ## add recurrent dropout into constants
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)
            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def step(self, inputs, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        dp_mask = states[2]
        rec_dp_mask = states[3]

        encode_dim = int(self.input_dim / 4)

        input_t = inputs[:, :encode_dim]
        input_p = inputs[:, encode_dim: encode_dim * 2]
        input_n = inputs[:, encode_dim * 2: encode_dim * 3]
        input_hs = inputs[:, encode_dim * 3:]

        x_i = K.dot(input_t * dp_mask[0], self.kernel_i) + self.bias_i
        x_f = K.dot(input_t * dp_mask[1], self.kernel_f) + self.bias_f
        x_c = K.dot(input_t * dp_mask[2], self.kernel_c) + self.bias_c
        x_o = K.dot(input_t * dp_mask[3], self.kernel_o) + self.bias_o

        i = self.recurrent_activation(x_i + K.dot(h_tm1 * rec_dp_mask[0],
                                                  self.recurrent_kernel_i))
        f = self.recurrent_activation(x_f + K.dot(h_tm1 * rec_dp_mask[1],
                                                  self.recurrent_kernel_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * rec_dp_mask[2],
                                                        self.recurrent_kernel_c))
        o = self.recurrent_activation(x_o + K.dot(h_tm1 * rec_dp_mask[3],
                                                  self.recurrent_kernel_o))
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, c]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(MulInput_LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
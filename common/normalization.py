"""
The implementations of Decorrelated Batch Normalization.
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables


class DecorelationNormalization(Layer):
    def __init__(self,
                 axis=-1,
                 momentum=0.9,
                 epsilon=1e-5,
                 m_per_group=0,
                 decomposition='zca_wm',
                 iter_num=5,
                 instance_norm=0,
                 data_format=None,
                 affine=True,
                 trainable=True,
                 name=None,
                 moving_mean_initializer='zeros',
                 moving_matrix_initializer='identity',
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 device=None,
                 **kwargs):
        assert decomposition in ['cholesky', 'zca', 'pca', 'iter_norm',
                                 'cholesky_wm', 'zca_wm', 'pca_wm', 'iter_norm_wm']
        super(DecorelationNormalization, self).__init__(name=name, trainable=trainable, **kwargs)
        self.axis = axis
        self.decomposition = decomposition
        self.iter_num = iter_num
        self.instance_norm = instance_norm
        self.device = device
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.momentum = momentum
        self.epsilon = epsilon
        self.m_per_group = m_per_group
        self.affine = affine
        self.trainable = trainable
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_projection_initializer = initializers.get(moving_matrix_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def matrix_initializer(self, shape, dtype=tf.float32, partition_info=None):
        moving_convs = []
        for i in range(shape[0]):
            moving_conv = tf.expand_dims(tf.eye(shape[1], dtype=dtype), 0)
            moving_convs.append(moving_conv)

        moving_convs = tf.concat(moving_convs, 0)
        return moving_convs

    def build(self, input_shape):
        assert self.data_format == 'channels_last'
        input_shape = input_shape.as_list()
        if self.axis < 0:
            self.axis = len(input_shape) + self.axis
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if self.m_per_group == 0:
            self.m_per_group = dim
        self.group = dim // self.m_per_group
        assert (dim % self.m_per_group == 0), 'dim is {}, m is {}'.format(dim, self.m_per_group)

        self.moving_mean = self.add_weight(shape=(dim, 1),
                                           name='moving_mean',
                                           synchronization=tf_variables.VariableSynchronization.ON_READ,
                                           initializer=self.moving_mean_initializer,
                                           trainable=False,
                                           aggregation=tf_variables.VariableAggregation.MEAN)
        self.moving_matrix = self.add_weight(shape=(self.group, self.m_per_group, self.m_per_group),
                                             name='moving_matrix',
                                             synchronization=tf_variables.VariableSynchronization.ON_READ,
                                             initializer=self.matrix_initializer,
                                             trainable=False,
                                             aggregation=tf_variables.VariableAggregation.MEAN)

        if self.affine:
            param_shape = [dim if i == self.axis
                           else 1 for i in range(len(input_shape))]
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False)
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.gamma = None
            self.beta = None

        self.built = True

    def call(self, inputs, training=None):
        shape = K.int_shape(inputs)
        if len(shape) == 4:
            w, h, c = shape[1:]
        elif len(shape) == 2:
            w, h, c = 1, 1, shape[-1]
            inputs = tf.expand_dims(inputs, 1)
            inputs = tf.expand_dims(inputs, 1)
        else:
            raise ValueError('shape not support:{}'.format(shape))

        bs = tf.shape(inputs)[0]

        m, f = center(inputs, self.moving_mean, w, h, c, self.instance_norm)
        get_inv_sqrt = get_decomposition(self.decomposition, bs, self.group, self.instance_norm, self.iter_num,
                                               self.epsilon, self.device)

        def train():
            ff_aprs = get_group_cov(f, self.group, self.m_per_group, self.instance_norm, bs, w, h, c)

            if self.instance_norm:
                ff_aprs = tf.transpose(ff_aprs, (1, 0, 2, 3))
                ff_aprs = (1 - self.epsilon) * ff_aprs + tf.expand_dims(tf.expand_dims(tf.eye(self.m_per_group) * self.epsilon, 0), 0)
            else:
                ff_aprs = (1 - self.epsilon) * ff_aprs + tf.expand_dims(tf.eye(self.m_per_group) * self.epsilon, 0)

            whitten_matrix = get_inv_sqrt(ff_aprs, self.m_per_group)[1]
            if self.instance_norm != 1:
                self.add_update([K.moving_average_update(self.moving_mean,
                                                         m,
                                                         self.momentum),
                                 K.moving_average_update(self.moving_matrix,
                                                         whitten_matrix if '_wm' in self.decomposition else ff_aprs,
                                                         self.momentum)],
                                inputs)
            return whitten_matrix

        def test():
            moving_matrix = (1 - self.epsilon) * self.moving_matrix + tf.eye(self.m_per_group) * self.epsilon
            if '_wm' in self.decomposition:
                return moving_matrix
            else:
                return get_inv_sqrt(moving_matrix, self.m_per_group)[1]

        if self.instance_norm == 1:
            inv_sqrt = train()
            f = tf.reshape(f, [-1, self.group, self.m_per_group, w*h])
            f_hat = tf.matmul(inv_sqrt, f)
            decorelated = K.reshape(f_hat, [bs, c, w, h])
            decorelated = tf.transpose(decorelated, [0, 2, 3, 1])
        else:
            inv_sqrt = K.in_train_phase(train, test, training=training)
            f = tf.reshape(f, [self.group, self.m_per_group, -1])
            f_hat = tf.matmul(inv_sqrt, f, name='whiten')
            decorelated = tf.reshape(f_hat, [c, bs, w, h])
            decorelated = tf.transpose(decorelated, [1, 2, 3, 0])

        if w == 1:
            decorelated = tf.squeeze(decorelated, 1)
        if h == 1:
            decorelated = tf.squeeze(decorelated, 1)
        if self.gamma is not None:
            scale = math_ops.cast(self.gamma, inputs.dtype)
            decorelated = decorelated * scale
        if self.beta is not None:
            offset = math_ops.cast(self.beta, inputs.dtype)
            decorelated = decorelated + offset
        return decorelated

    def compute_output_shape(self, input_shape):
        return input_shape


def center(inputs, moving_mean, w, h, c, instance_norm=False):
    if instance_norm:
        x_t = tf.transpose(inputs, (0, 3, 1, 2))
        x_flat = tf.reshape(x_t, (-1, c, w * h))
        # (bs, c, w*h)
        m = tf.reduce_mean(x_flat, axis=2, keepdims=True)
        # (bs, c, 1)
    else:
        x_t = tf.transpose(inputs, (3, 0, 1, 2))
        x_flat = tf.reshape(x_t, (c, -1))
        # (c, bs*w*h)
        m = tf.reduce_mean(x_flat, axis=1, keepdims=True)
        m = K.in_train_phase(m, moving_mean)
        # (c, 1)
    f = x_flat - m
    return m, f


def get_decomposition(decomposition, batch_size, group, instance_norm, iter_num, epsilon, device=None):
    if device == 'cpu':
        device = '/cpu:0'
    elif device == 'gpu':
        device = '/gpu:0'
    if decomposition == 'cholesky' or decomposition == 'cholesky_wm':
        if device is None:
            device = '/cpu:0'

        def get_inv_sqrt(ff, m_per_group):
            with tf.device(device):
                sqrt = tf.linalg.cholesky(ff)
            if instance_norm:
                inv_sqrt = tf.linalg.triangular_solve(sqrt,
                                                      tf.tile(tf.expand_dims(tf.expand_dims(tf.eye(m_per_group), 0), 0),
                                                              [batch_size, group, 1, 1]))
            else:
                inv_sqrt = tf.linalg.triangular_solve(sqrt, tf.tile(tf.expand_dims(tf.eye(m_per_group), 0),
                                                                    [group, 1, 1]))
            return sqrt, inv_sqrt
    elif decomposition == 'zca' or decomposition == 'zca_wm':
        if device is None:
            device = '/cpu:0'

        def get_inv_sqrt(ff, m_per_group):
            with tf.device(device):
                S, U, _ = tf.linalg.svd(ff + tf.eye(m_per_group) * epsilon, full_matrices=True)
            D = tf.linalg.diag(tf.pow(S, -0.5))
            inv_sqrt = tf.matmul(tf.matmul(U, D), U, transpose_b=True)
            D = tf.linalg.diag(tf.pow(S, 0.5))
            sqrt = tf.matmul(tf.matmul(U, D), U, transpose_b=True)
            return sqrt, inv_sqrt
    elif decomposition == 'pca' or decomposition == 'pca_wm':
        if device is None:
            device = '/cpu:0'

        def get_inv_sqrt(ff, m_per_group):
            with tf.device(device):
                S, U, _ = tf.linalg.svd(ff + tf.eye(m_per_group) * epsilon, full_matrices=True)
            D = tf.linalg.diag(tf.pow(S, -0.5))
            inv_sqrt = tf.matmul(D, U, transpose_b=True)
            D = tf.linalg.diag(tf.pow(S, 0.5))
            sqrt = tf.matmul(D, U, transpose_b=True)
            return sqrt, inv_sqrt
    elif decomposition == 'iter_norm' or decomposition == 'iter_norm_wm':
        if device is None:
            device = '/gpu:0'

        def get_inv_sqrt(ff, m_per_group):
            trace = tf.linalg.trace(ff)
            trace = tf.expand_dims(trace, [-1])
            trace = tf.expand_dims(trace, [-1])
            sigma_norm = ff / trace

            projection = tf.eye(m_per_group)
            projection = tf.expand_dims(projection, 0)
            projection = tf.tile(projection, [group, 1, 1])
            for i in range(iter_num):
                projection = (3 * projection - tf.matmul(tf.matmul(tf.matmul(projection, projection), projection), sigma_norm)) / 2

            return None, projection / tf.sqrt(trace)
    else:
        assert False
    return get_inv_sqrt


def get_group_cov(inputs, group, m_per_group, instance_norm, bs, w, h, c):
    ff_aprs = []
    for i in range(group):
        start_index = i * m_per_group
        end_index = np.min(((i + 1) * m_per_group, c))
        if instance_norm:
            centered = inputs[:, start_index:end_index, :]
        else:
            centered = inputs[start_index:end_index, :]
        ff_apr = tf.matmul(centered, centered, transpose_b=True)
        ff_apr = tf.expand_dims(ff_apr, 0)
        ff_aprs.append(ff_apr)

    ff_aprs = tf.concat(ff_aprs, 0)
    if instance_norm:
        ff_aprs /= (tf.cast(w * h, tf.float32) - 1.)
    else:
        ff_aprs /= (tf.cast(bs * w * h, tf.float32) - 1.)
    return ff_aprs


def test_decorrectedBN():
    data = tf.random.normal(shape=[256, 5])
    sigma = tf.matmul(tf.linalg.matrix_transpose(data), data)
    s, u, vt = tf.linalg.svd(sigma)
    result = tf.linalg.matmul(tf.linalg.matmul(u, tf.linalg.diag(s)), tf.linalg.matrix_transpose(vt))

    y2 = DecorelationNormalization(affine=False)(data, True)
    y2_inference = DecorelationNormalization(affine=False)(data, False)
    y2_sigma = tf.matmul(tf.linalg.matrix_transpose(y2), y2) / 256
    s2, u2, v2 = tf.linalg.svd(y2_sigma)
    print(s2)
    print()
    print(y2_sigma)
    print()
    import matplotlib.pyplot as mp, seaborn

    seaborn.heatmap(y2_sigma, center=0, annot=True)
    mp.show()


def test_cnn_dbn():
    import matplotlib.pyplot as mp
    import seaborn
    data = tf.random.normal(shape=[2, 4, 4, 16])
    data = tf.concat([data, data, data, data, data, data, data, data], 0)
    x = tf.reshape(data, [256, 16])
    x_sigma = tf.matmul(tf.linalg.matrix_transpose(x), x) / 256
    print()
    seaborn.heatmap(x_sigma, center=0, annot=True)
    mp.show()

    y = DecorelationNormalization(m_per_group=0, affine=False)(data, True)
    y = DecorelationNormalization(m_per_group=0, affine=False)(data, False)
    y_ = tf.reshape(y, [256, 16])
    y_sigma = tf.matmul(tf.linalg.matrix_transpose(y_), y_) / 256
    print()
    seaborn.heatmap(y_sigma, center=0, annot=True)
    mp.show()


def test_whitten():
    data = tf.random.normal(shape=[256, 16])
    sigma = tf.matmul(tf.linalg.matrix_transpose(data), data)
    s, u, v = tf.linalg.svd(sigma)
    y = DecorelationNormalization(affine=False, decomposition='iter_norm', iter_num=5)(data, True)
    y_sigma = tf.matmul(tf.linalg.matrix_transpose(y), y) / 256
    s1, u1, v1 = tf.linalg.svd(y_sigma)

    y2 = DecorelationNormalization(affine=False)(data, True)
    y2_sigma = tf.matmul(tf.linalg.matrix_transpose(y2), y2) / 256
    s2, u2, v2 = tf.linalg.svd(y2_sigma)
    print(s1)
    print(s2)
    print()
    print(y_sigma)
    print(y2_sigma)
    print()





import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.keras.layers import Layer
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.keras.utils import conv_utils

sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./gan'))


class DecorelationNormalization(Layer):
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 m_per_group=0,
                 decomposition='cholesky',
                 iter_num=5,
                 instance_norm=0,
                 renorm=False,
                 data_format=None,
                 moving_mean_initializer='zeros',
                 moving_cov_initializer='identity',
                 device='cpu',
                 **kwargs):
        assert decomposition in ['cholesky', 'zca', 'pca', 'iter_norm',
                                 'cholesky_wm', 'zca_wm', 'pca_wm', 'iter_norm_wm']
        super(DecorelationNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.momentum = momentum
        self.epsilon = epsilon
        self.m_per_group = m_per_group
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        # self.moving_cov_initializer = initializers.get(moving_cov_initializer)
        self.axis = axis
        self.renorm = renorm
        self.decomposition = decomposition
        self.iter_num = iter_num
        self.instance_norm = instance_norm
        self.device = device
        self.data_format = conv_utils.normalize_data_format(data_format)

    def matrix_initializer(self, shape, dtype=tf.float32, partition_info=None):
        moving_convs = []
        for i in range(shape[0]):
            moving_conv = tf.expand_dims(tf.eye(shape[1], dtype=dtype), 0)
            moving_convs.append(moving_conv)

        moving_convs = tf.concat(moving_convs, 0)
        return moving_convs

    def build(self, input_shape):
        assert self.data_format == 'channels_last'
        dim = input_shape.as_list()[self.axis]
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

        self.built = True

    def call(self, inputs, training=None):
        _, w, h, c = K.int_shape(inputs)
        bs = K.shape(inputs)[0]

        m, f = utils.center(inputs, self.moving_mean, self.instance_norm)
        get_inv_sqrt = utils.get_decomposition(self.decomposition, bs, self.group, self.instance_norm, self.iter_num,
                                               self.epsilon, self.device)

        def train():
            ff_aprs = utils.get_group_cov(f, self.group, self.m_per_group, self.instance_norm, bs, w, h, c)

            if self.instance_norm:
                ff_aprs = tf.transpose(ff_aprs, (1, 0, 2, 3))
                ff_aprs = (1 - self.epsilon) * ff_aprs + tf.expand_dims(tf.expand_dims(tf.eye(self.m_per_group) * self.epsilon, 0), 0)
            else:
                ff_aprs = (1 - self.epsilon) * ff_aprs + tf.expand_dims(tf.eye(self.m_per_group) * self.epsilon, 0)

            whitten_matrix = get_inv_sqrt(ff_aprs, self.m_per_group)[1]

            self.add_update([K.moving_average_update(self.moving_mean,
                                                     m,
                                                     self.momentum),
                             K.moving_average_update(self.moving_matrix,
                                                     whitten_matrix if '_wm' in self.decomposition else ff_aprs,
                                                     self.momentum)],
                            inputs)

            if self.renorm:
                l, l_inv = get_inv_sqrt(ff_aprs, self.m_per_group)
                ff_mov = (1 - self.epsilon) * self.moving_matrix + tf.eye(self.m_per_group) * self.epsilon
                _, l_mov_inverse = get_inv_sqrt(ff_mov, self.m_per_group)
                l_ndiff = K.stop_gradient(l)
                return tf.matmul(tf.matmul(l_mov_inverse, l_ndiff), l_inv)

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
            inv_sqrt = K.in_train_phase(train, test)
            f = tf.reshape(f, [self.group, self.m_per_group, -1])
            f_hat = tf.matmul(inv_sqrt, f)
            decorelated = K.reshape(f_hat, [c, bs, w, h])
            decorelated = tf.transpose(decorelated, [1, 2, 3, 0])

        return decorelated

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_matrix_initializer': initializers.serialize(self.matrix_initializer)
        }
        base_config = super(DecorelationNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def test_dbn_eager():
    tf.enable_eager_execution()
    data = tf.random.normal([1, 7, 7, 16])
    K.set_learning_phase(1)
    # tf.set_random_seed(1)
    decor1 = DecorelationNormalization(m_per_group=2, decomposition='zca', instance_norm=1)
    decor2 = DecorelationNormalization(m_per_group=2, decomposition='zca', instance_norm=0)
    decor3 = DecorelationNormalization(m_per_group=2, decomposition='zca', instance_norm=0, group_conv=2)
    out1 = decor1(data)
    out2 = decor2(data)
    out3 = decor3(data)
    out1 = tf.reduce_sum(out1)
    out2 = tf.reduce_sum(out2)
    out3 = tf.reduce_sum(out3)
    print(out1)
    print(out2)
    print(out3)


def test_dbn_eager2():
    tf.enable_eager_execution()
    data = tf.random.normal([1, 7, 7, 16])
    K.set_learning_phase(1)
    # tf.set_random_seed(1)
    decor1 = DecorelationNormalization(m_per_group=2, decomposition='pca', instance_norm=0)
    decor2 = DecorelationNormalization(m_per_group=2, decomposition='pca', instance_norm=1)
    import time
    out1 = decor1(data)
    t1 = time.time()
    out1 = decor1(data)
    t2 = time.time()
    out2 = decor2(data)
    t3 = time.time()
    distance = np.sum(np.square(out1-out2))
    print(distance)
    print('t1:', t2 - t1)
    print('t2:', t3 - t2)


def test_dbn2():
    # tf.enable_eager_execution()
    inputs = tf.keras.Input((7, 7, 16))
    data = np.random.normal(0, 1, [1, 7, 7, 16])
    data2 = np.random.normal(0, 1, [256, 7, 7, 16])
    K.set_learning_phase(1)
    # tf.set_random_seed(1)
    decor1 = DecorelationNormalization(m_per_group=2, decomposition='zca', instance_norm=0)
    decor2 = DecorelationNormalization(m_per_group=2, decomposition='zca', instance_norm=1)
    out1, out2 = decor1(inputs), decor2(inputs)
    op1 = K.function(inputs, out1)
    op2 = K.function(inputs, out2)
    import time
    out2 = op2(data2)
    t1 = time.time()
    out1 = op1(data)
    t2 = time.time()
    out2 = op2(data)
    t3 = time.time()
    distance = np.sum(np.square(out1-out2))
    print(distance)
    print('t1:', t2 - t1)
    print('t2:', t3 - t2)


def test_dbn():
    inputs = tf.keras.layers.Input([8, 8, 16])
    data = np.random.normal(0, 1, [128, 8, 8, 16])
    # K.set_learning_phase(1)
    decor = DecorelationNormalization(group=1, instance_norm=1)
    out = decor(inputs)
    out = tf.reduce_mean(out)
    sess = K.get_session()
    sess.run(tf.global_variables_initializer())
    K.set_learning_phase(1)
    outputs = sess.run([out], feed_dict={inputs: data})
    print(np.mean(outputs))


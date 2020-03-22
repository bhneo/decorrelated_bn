import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K


class TensorLog(object):
    def __init__(self):
        self.hist = {}
        self.scalar = {}
        self.image = {}
        self.tensor = {}
        self.model = None

    def add_hist(self, name, tensor):
        self.hist[name] = tensor

    def add_scalar(self, name, tensor):
        self.scalar[name] = tensor

    def add_image(self, name, image):
        self.image[name] = image

    def add_tensor(self, name, tensor):
        self.tensor[name] = tensor

    def get_outputs(self):
        outputs = []
        for key in self.hist:
            outputs.append(self.hist[key])
        for key in self.scalar:
            outputs.append(self.scalar[key])
        for key in self.image:
            outputs.append(self.image[key])
        for key in self.tensor:
            outputs.append(self.tensor[key])
        return outputs

    def set_model(self, model):
        self.model = model

    def summary(self, outputs, epoch):
        i = 0
        for key in self.hist:
            tf.summary.histogram(key, outputs[i], step=epoch + 1)
            i += 1
        for key in self.scalar:
            tf.summary.scalar(key, outputs[i], step=epoch + 1)
            i += 1
        for key in self.image:
            tf.summary.image(key, outputs[i], step=epoch + 1)
            i += 1
        for key in self.tensor:
            i += 1


def str2bool(obj):
    if isinstance(obj, str):
        if obj == 'True':
            return True
        elif obj == 'False':
            return False
        else:
            raise TypeError('Type not support:{}'.format(obj))
    if isinstance(obj, bool):
        return obj
    else:
        raise TypeError('{} is not str'.format(obj))


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
                projection = (3 * projection - projection * projection * projection * sigma_norm) / 2

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



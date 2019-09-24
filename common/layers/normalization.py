"""
The implementations of Decorrelated Batch Normalization, including DecorrelatedBN and DecorrelatedBNPowerIter.
"""

import matplotlib.pyplot as mp
import numpy as np
import seaborn
import time
import tensorflow as tf

from config import config as cfg

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables


class DecorrelatedBN(Layer):
    _USE_V2_BEHAVIOR = True

    def __init__(self,
                 axis=-1,
                 momentum=0.9,
                 epsilon=1e-5,
                 m_per_group=0,
                 affine=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_projection_initializer='identity',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):

        super(DecorrelatedBN, self).__init__(name=name, trainable=trainable, **kwargs)
        if isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError('axis must be int, type given: %s'
                            % type(self.axis))
        self.momentum = momentum
        self.epsilon = epsilon
        self.m_per_group = m_per_group
        self.affine = affine
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_projection_initializer = initializers.get(moving_projection_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        self.moving_means = []
        self.moving_projections = []

        self._trainable_var = None

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        if self._trainable_var is not None:
            self._trainable_var.update_value(value)

    def _get_trainable_var(self):
        if self._trainable_var is None:
            self._trainable_var = K.freezable_variable(
                self._trainable, name=self.name + '_trainable')
        return self._trainable_var

    def _get_training_value(self, training=None):
        if training is None:
            training = K.learning_phase()
        if self._USE_V2_BEHAVIOR:
            if isinstance(training, int):
                training = bool(training)
            if base_layer_utils.is_in_keras_graph():
                training = math_ops.logical_and(training, self._get_trainable_var())
            else:
                training = math_ops.logical_and(training, self.trainable)
        return training

    @property
    def _param_dtype(self):
        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
            return dtypes.float32
        else:
            return self.dtype or dtypes.float32

    def build(self, input_shape):
        # assert (len(input_shape) != 2 or len(input_shape) != 4), \
        #     'only 4D or 2D tensor supported, got {}D tensor instead'.format(len(input_shape))
        input_shape = tensor_shape.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank:', input_shape)
        ndims = len(input_shape)

        if self.axis < 0:
            self.axis = ndims + self.axis

        axis_to_dim = {self.axis: input_shape.dims[self.axis].value}
        if axis_to_dim[self.axis] is None:
            raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                             input_shape)
        if self.m_per_group == 0 or self.m_per_group > axis_to_dim[self.axis]:
            self.m_per_group = axis_to_dim[self.axis]

        self.input_spec = InputSpec(ndim=ndims, axes=axis_to_dim)

        # allow channels % m_per_group != 0
        self.groups = int(np.floor((axis_to_dim[self.axis] - 1) / self.m_per_group) + 1)
        dim = axis_to_dim[self.axis]
        for i in range(0, self.groups):
            mean_name = 'moving_mean{}'.format(i)
            projection_name = 'moving_projection{}'.format(i)
            if i < self.groups - 1:
                mean_shape = [1, self.m_per_group]
                projection_shape = [self.m_per_group, self.m_per_group]
            else:
                mean_shape = [1, dim - (self.groups - 1) * self.m_per_group]
                projection_shape = [dim - (self.groups - 1) * self.m_per_group, dim - (self.groups - 1) * self.m_per_group]

            moving_mean = self.add_weight(
                name=mean_name,
                shape=mean_shape,
                dtype=self._param_dtype,
                initializer=self.moving_mean_initializer,
                synchronization=tf_variables.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf_variables.VariableAggregation.MEAN,
                experimental_autocast=False)
            moving_projection = self.add_weight(
                name=projection_name,
                shape=projection_shape,
                dtype=self._param_dtype,
                initializer=self.moving_projection_initializer,
                synchronization=tf_variables.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf_variables.VariableAggregation.MEAN,
                experimental_autocast=False)

            self.moving_means.append(moving_mean)
            self.moving_projections.append(moving_projection)

        if self.affine:
            param_shape = [axis_to_dim[i] if i in axis_to_dim
                           else 1 for i in range(ndims)]
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False)
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False)
        else:
            self.gamma = None
            self.beta = None

        super(DecorrelatedBN, self).build(input_shape)

    def call(self, inputs, training=None):
        training = self._get_training_value(training)

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in [self.axis]]

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis] = input_shape.dims[self.axis].value

        def _broadcast(v):
            if (v is not None and len(v.get_shape()) != ndims and
                    reduction_axes != list(range(ndims - 1))):
                return array_ops.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)
        if self.axis != ndims-1:
            trans = reduction_axes+[self.axis]
            transpose_recover = [i for i in range(self.axis)] + [ndims-1] + [j for j in range(self.axis, ndims-1)]
            inputs = array_ops.transpose(inputs, perm=trans)
            transposed_shape = [-1] + inputs.get_shape().as_list()[1:]

        inputs = array_ops.reshape(inputs, shape=[-1, input_shape.dims[self.axis].value])

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = tf_utils.constant_value(training)
        outputs = []
        height, width = input_shape.as_list()[1], input_shape.as_list()[2]
        for i in range(self.groups):
            start_index = i * self.m_per_group
            end_index = np.min(((i + 1) * self.m_per_group, input_shape.dims[self.axis].value))
            group_input = inputs[:, start_index:end_index]

            if training_value is not False:
                mean = tf.reduce_mean(group_input, 0, keepdims=True)
                centered = group_input - mean

                # centered_ = tf.expand_dims(centered, -1)
                # sigma = tf.matmul(centered_, tf.linalg.matrix_transpose(centered_))
                # sigma = tf.reduce_mean(sigma, 0)

                sigma = tf.matmul(tf.linalg.matrix_transpose(centered), centered)
                sigma /= (cfg.training.batch_size*height*width)

                projection = self.get_projection(sigma, group_input)

                moving_mean = self.moving_means[i]
                moving_projection = self.moving_projections[i]

                mean = tf_utils.smart_cond(training,
                                           lambda: mean,
                                           lambda: ops.convert_to_tensor(moving_mean))
                projection = tf_utils.smart_cond(training,
                                                 lambda: projection,
                                                 lambda: ops.convert_to_tensor(moving_projection))

                new_mean, new_projection = mean, projection

                def _do_update(var, value):
                    return self._assign_moving_average(var, value, self.momentum, None)

                def mean_update():
                    true_branch = lambda: _do_update(self.moving_means[i], new_mean)
                    false_branch = lambda: self.moving_means[i]
                    return tf_utils.smart_cond(training, true_branch, false_branch)

                def projection_update():
                    true_branch = lambda: _do_update(self.moving_projections[i], new_projection)
                    false_branch = lambda: self.moving_projections[i]
                    return tf_utils.smart_cond(training, true_branch, false_branch)

                self.add_update(mean_update)
                self.add_update(projection_update)

            else:
                mean, projection = self.moving_means[i], self.moving_projections[i]
                centered = group_input - mean

            mean = math_ops.cast(mean, inputs.dtype)
            projection = math_ops.cast(projection, inputs.dtype)

            output = tf.matmul(centered, projection)
            outputs.append(output)

        outputs = tf.concat(outputs, 1)
        if self.axis != ndims - 1:
            outputs = tf.reshape(outputs, shape=transposed_shape)
            outputs = tf.transpose(outputs, perm=transpose_recover)
        else:
            outputs = tf.reshape(outputs, shape=[-1]+input_shape.as_list()[1:])

        if scale is not None:
            scale = math_ops.cast(scale, inputs.dtype)
            outputs = outputs * scale
        if offset is not None:
            offset = math_ops.cast(offset, inputs.dtype)
            outputs += offset

        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)
        return outputs

    def get_projection(self, sigma, inputs):
        eig, rotation, _ = tf.linalg.svd(sigma)
        eig += self.epsilon
        eig = tf.pow(eig, -1 / 2)
        eig = tf.linalg.diag(eig)

        whitten_matrix = tf.matmul(rotation, eig)
        return tf.matmul(whitten_matrix, tf.transpose(rotation))

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        with K.name_scope('AssignMovingAvg') as scope:
            with ops.colocate_with(variable):
                decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
                if decay.dtype != variable.dtype.base_dtype:
                    decay = math_ops.cast(decay, variable.dtype.base_dtype)
                update_delta = (variable - math_ops.cast(value, variable.dtype)) * decay
                if inputs_size is not None:
                    update_delta = array_ops.where(inputs_size > 0, update_delta,
                                                   K.zeros_like(update_delta))
                return state_ops.assign_sub(variable, update_delta, name=scope)

    def compute_output_shape(self, input_shape):
        return input_shape


class IterativeNormalization(DecorrelatedBN):
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-5,
                 m_per_group=0,
                 affine=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_projection_initializer='identity',
                 iter_num=5,
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):

        super(IterativeNormalization, self).__init__(axis,
                                                     momentum,
                                                     epsilon, m_per_group,
                                                     affine,
                                                     beta_initializer,
                                                     gamma_initializer,
                                                     moving_mean_initializer,
                                                     moving_projection_initializer,
                                                     beta_regularizer,
                                                     gamma_regularizer,
                                                     beta_constraint,
                                                     gamma_constraint,
                                                     trainable, name,
                                                     **kwargs)
        self.iter_num = iter_num

    def get_projection(self, sigma, inputs):
        n_feature = inputs.get_shape().dims[-1].value
        trace = tf.linalg.trace(sigma)
        sigma_norm = sigma / trace

        projection = tf.eye(n_feature)
        # x1 = projection * projection
        # x2 = tf.matmul(projection, projection)
        for i in range(self.iter_num):
            projection = (3 * projection - projection * projection * projection * sigma_norm) / 2

        return projection / tf.sqrt(trace)


def test_BN():
    inputs = tf.random.normal([128, 32, 32, 3])
    outputs = tf.keras.layers.BatchNormalization()(inputs)


def test_decorrectedBN():
    data = tf.random.normal(shape=[256, 5])
    sigma = tf.matmul(tf.linalg.matrix_transpose(data), data)
    s, u, vt = tf.linalg.svd(sigma)
    result = tf.linalg.matmul(tf.linalg.matmul(u, tf.linalg.diag(s)), tf.linalg.matrix_transpose(vt))

    y2 = DecorrelatedBN(affine=False)(data, True)
    y2_inference, pro2_inference = DecorrelatedBN(affine=False)(data, False)
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
    data = tf.random.normal(shape=[2, 4, 4, 16])
    data = tf.concat([data, data, data, data, data, data, data, data], 0)
    x = tf.reshape(data, [256, 16])
    x_sigma = tf.matmul(tf.linalg.matrix_transpose(x), x) / 256
    print()
    seaborn.heatmap(x_sigma, center=0, annot=True)
    mp.show()

    y = DecorrelatedBN(m_per_group=8)(data, True)
    y_ = tf.reshape(y, [256, 16])
    y_sigma = tf.matmul(tf.linalg.matrix_transpose(y_), y_) / 256
    print()
    seaborn.heatmap(y_sigma, center=0, annot=True)
    mp.show()


def test_whitten():
    data = tf.random.normal(shape=[256, 16])
    sigma = tf.matmul(tf.linalg.matrix_transpose(data), data)
    s, u, v = tf.linalg.svd(sigma)
    y = IterativeNormalization(affine=False, iter_num=5)(data, True)
    y_sigma = tf.matmul(tf.linalg.matrix_transpose(y), y) / 256
    s1, u1, v1 = tf.linalg.svd(y_sigma)

    y2 = DecorrelatedBN(affine=False)(data, True)
    y2_sigma = tf.matmul(tf.linalg.matrix_transpose(y2), y2) / 256
    s2, u2, v2 = tf.linalg.svd(y2_sigma)
    print(s1)
    print(s2)
    print()
    print(y_sigma)
    print(y2_sigma)
    print()


def test_speed():
    data = tf.random.normal(shape=[100, 128, 128])
    @tf.function
    def func(data):
        t1 = time.time()
        s1, u1, v1 = tf.linalg.svd(data)
        t2 = time.time()
        print('t1:', t2-t1)

        s2, u2, v2 = [], [], []
        for i in range(100):
            s, u, v = tf.linalg.svd(tf.expand_dims(data[i], 0))
            s2.append(s)
            u2.append(u)
            v2.append(v)
        t3 = time.time()
        print('t2:', t3-t2)

        s2 = tf.concat(s2, 0)
        u2 = tf.concat(u2, 0)
        v2 = tf.concat(v2, 0)

        print(tf.reduce_sum(s1 - s2))
        print(tf.reduce_sum(u1 - u2))
        print(tf.reduce_sum(v1 - v2))

    func(data)


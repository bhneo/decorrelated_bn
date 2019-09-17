"""
The implementations of Decorrelated Batch Normalization, including DecorrelatedBN and DecorrelatedBNPowerIter.
"""

import contextlib

import numpy as np
import tensorflow as tf
from config import cfg
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables


class DecorrelatedBN(Layer):
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

    def build(self, input_shape):
        assert (len(input_shape) != 2 or len(input_shape) != 4), \
            'only 4D or 2D tensor supported, got {}D tensor instead'.format(len(input_shape))
        input_shape = tensor_shape.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank:', input_shape)
        ndims = len(input_shape)

        if self.axis < 0:
            self.axis = ndims + self.axis

        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == dtypes.float16 or self.dtype == dtypes.bfloat16:
            param_dtype = dtypes.float32
        else:
            param_dtype = self.dtype or dtypes.float32

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
            projection_name = 'moving_variance{}'.format(i)
            if i < self.groups - 1:
                mean_shape = [1, self.m_per_group]
                projection_shape = [self.m_per_group, self.m_per_group]
            else:
                mean_shape = [1, dim - (self.groups - 1) * self.m_per_group]
                projection_shape = [dim - (self.groups - 1) * self.m_per_group, dim - (self.groups - 1) * self.m_per_group]

            moving_mean = self.add_weight(
                name=mean_name,
                shape=mean_shape,
                dtype=param_dtype,
                initializer=self.moving_mean_initializer,
                synchronization=tf_variables.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf_variables.VariableAggregation.MEAN)
            moving_projection = self.add_weight(
                name=projection_name,
                shape=projection_shape,
                dtype=param_dtype,
                initializer=self.moving_projection_initializer,
                synchronization=tf_variables.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf_variables.VariableAggregation.MEAN)

            self.moving_means.append(moving_mean)
            self.moving_projections.append(moving_projection)

        if self.affine:
            param_shape = [axis_to_dim[i] if i in axis_to_dim
                           else 1 for i in range(ndims)]
            self.gamma = self.add_weight(
                name='gamma',
                shape=param_shape,
                dtype=param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True)
            self.beta = self.add_weight(
                name='beta',
                shape=param_shape,
                dtype=param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True)
        else:
            self.gamma = None
            self.beta = None

        super(DecorrelatedBN, self).build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        in_eager_mode = context.executing_eagerly()
        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.get_shape()
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in [self.axis]]

        trans = reduction_axes+[self.axis]
        trans_recover = [i for i in range(self.axis)] + [ndims-1] + [j for j in range(self.axis, ndims-1)]
        inputs = array_ops.transpose(inputs, perm=trans)
        transposed_shape = [-1] + inputs.get_shape().as_list()[1:]
        inputs = array_ops.reshape(inputs, shape=[-1, input_shape.dims[self.axis].value])

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = tf_utils.constant_value(training)
        outputs = []
        for i in range(self.groups):
            start_index = i * self.m_per_group
            end_index = np.min(((i + 1) * self.m_per_group, input_shape.dims[self.axis].value))
            group_input = inputs[:, start_index:end_index]

            if training_value is not False:
                mean = tf.reduce_mean(group_input, 0, keepdims=True)
                centered = group_input - mean

                # centered_ = tf.expand_dims(centered, -1)
                #
                # sigma = tf.matmul(centered_, tf.matrix_transpose(centered_))
                # sigma = tf.reduce_mean(sigma, 0)

                sigma = tf.matmul(tf.matrix_transpose(centered), centered)
                sigma = sigma * 1 / cfg.batch_size
                projection = self.get_projection(sigma, group_input)

                moving_mean = self.moving_means[i]
                moving_projection = self.moving_projections[i]

                mean = tf_utils.smart_cond(training,
                                           lambda: mean,
                                           lambda: moving_mean)
                projection = tf_utils.smart_cond(training,
                                                 lambda: projection,
                                                 lambda: moving_projection)

                new_mean, new_projection = mean, projection

                def _do_update(var, value):
                    if in_eager_mode and not self.trainable:
                        return

                    return self._assign_moving_average(var, value, self.momentum)

                mean_update = tf_utils.smart_cond(
                    training,
                    lambda: _do_update(self.moving_means[i], new_mean),
                    lambda: self.moving_means[i])
                projection_update = tf_utils.smart_cond(
                    training,
                    lambda: _do_update(self.moving_projections[i], new_projection),
                    lambda: self.moving_projections[i])
                if not context.executing_eagerly():
                    self.add_update(mean_update, inputs=True)
                    self.add_update(projection_update, inputs=True)

            else:
                mean, projection = self.moving_means[i], self.moving_projections[i]
                centered = group_input - mean

            mean = math_ops.cast(mean, inputs.dtype)
            projection = math_ops.cast(projection, inputs.dtype)

            output = tf.matmul(centered, projection)
            outputs.append(output)

        outputs = tf.concat(outputs, 1)
        outputs = tf.reshape(outputs, shape=transposed_shape)
        outputs = tf.transpose(outputs, perm=trans_recover)

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis] = input_shape.dims[self.axis].value

        def _broadcast(v):
            if (v is not None and
                    len(v.get_shape()) != ndims and
                    reduction_axes != list(range(ndims - 1))):
                return array_ops.reshape(v, broadcast_shape)
            return v
        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)
        if offset is not None:
            offset = math_ops.cast(offset, inputs.dtype)

        if self.affine:
            outputs = outputs * scale + offset

        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)
        return outputs, projection

    def get_projection(self, sigma, inputs):
        eig, rotation, _ = tf.svd(sigma)
        eig += self.epsilon
        eig = tf.pow(eig, -1 / 2)
        eig = tf.diag(eig)

        whitten_matrix = tf.matmul(rotation, eig)
        return tf.matmul(whitten_matrix, tf.transpose(rotation))

    def _assign_moving_average(self, variable, value, momentum):
        with ops.name_scope(None, 'AssignMovingAvg',
                            [variable, value, momentum]) as scope:
            # TODO(apassos,srbs,skyewm): the colocation constraints here are disabled
            # because of a bug which leads cond_v2 to skip rewriting them creating
            # conflicts.
            if tf2.enabled():
                cm = contextlib.contextmanager(lambda: (yield))()
            else:
                cm = ops.colocate_with(variable)
            with cm:
                decay = ops.convert_to_tensor(1.0 - momentum, name='decay')
                if decay.dtype != variable.dtype.base_dtype:
                    decay = math_ops.cast(decay, variable.dtype.base_dtype)
                update_delta = (variable - math_ops.cast(value, variable.dtype)) * decay
                return state_ops.assign_sub(variable, update_delta, name=scope)

    def compute_output_shape(self, input_shape):
        return input_shape

    def updateOutput_perGroup_train(self, data, groupId):
        mean = tf.reduce_mean(data, 0)
        update_means = tf.assign(self.moving_means[groupId],
                                 self.moving_means[groupId] * (1 - self.momentum) + mean * self.momentum)

        centered = tf.expand_dims(data - mean, -1)
        # self.summaries.append(tf.summary.histogram('group{}_centered'.format(groupId), centered))

        sigma = tf.matmul(centered, tf.matrix_transpose(centered))
        sigma = tf.reduce_mean(sigma, 0)

        eig, rotation, _ = tf.svd(sigma)
        eig += self.eps
        eig = tf.pow(eig, -1/2)
        eig = tf.diag(eig)

        whitten_matrix = tf.matmul(rotation, eig)
        whitten_matrix = tf.matmul(whitten_matrix, tf.transpose(rotation))

        update_projections = tf.assign(self.running_projections[groupId],
                                 self.running_projections[groupId] * (1 - self.momentum) + whitten_matrix * self.momentum)

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_means)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_projections)
        return tf.matmul(tf.squeeze(centered), whitten_matrix)


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
        trace = tf.trace(sigma)
        sigma_norm = sigma / trace

        projection = tf.eye(n_feature)
        x1 = projection * projection
        x2 = tf.matmul(projection, projection)
        for i in range(self.iter_num):
            projection = (3 * projection - projection * projection * projection * sigma_norm) / 2

        return projection / tf.sqrt(trace)



if __name__ == "__main__":
    tf.enable_eager_execution()
    data = tf.random_normal(shape=[256, 16])
    sigma = tf.matmul(tf.matrix_transpose(data), data)
    s, u, v = tf.svd(sigma)
    y,pro1 = IterativeNormalization(affine=False, iter_num=5)(data, True)
    y_sigma = tf.matmul(tf.matrix_transpose(y), y)/256
    s1, u1, v1 = tf.svd(y_sigma)

    y2,pro2 = DecorrelatedBN(affine=False)(data, True)
    y2_sigma = tf.matmul(tf.matrix_transpose(y2), y2)/256
    s2, u2, v2 = tf.svd(y2_sigma)
    print(s1)
    print(s2)
    print()
    print(y_sigma)
    print(y2_sigma)
    print()
    gap = pro1-pro2
    print(gap)

    import matplotlib.pyplot as mp, seaborn

    seaborn.heatmap(gap, center=0, annot=True)
    mp.show()
    #
    # seaborn.heatmap(y_sigma, center=0, annot=True)
    # mp.show()
    # seaborn.heatmap(y2_sigma, center=0, annot=True)
    # mp.show()

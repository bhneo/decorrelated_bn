"""
The Basic Decorelated Batch normalization version, in which:
(1) use ZCA to whitening the activation
(2) include train mode and test mode. in training mode, we train the module
"""
import numpy as np
import tensorflow as tf


def buildDBN(inputs, train, affine=True):
    input_shape = inputs.get_shape().as_list()
    DBN = DecorelateBNPowerIter(input_shape, affine=affine)
    return DBN.updateOutput(inputs, train)


class DecorelateBNPowerIter:
    def __init__(self, shape, m_perGroup=0, affine=False, nIter=5, momentum=0.1, data_format='channels_last'):
        assert (len(shape) != 2 or len(shape) != 4), \
            'only 4D or 2D tensor supported, got {}D tensor instead'.format(len(shape))
        self.input_shape = shape
        if len(shape) == 4:
            self.data_format = data_format
            if data_format == 'channels_last':
                self.nDim = shape[3]
                self.iH = shape[1]
                self.iW = shape[2]
            else:
                self.nDim = shape[1]
                self.iH = shape[2]
                self.iW = shape[3]
        else:
            self.nDim = shape[1]

        self.affine = affine

        if m_perGroup == 0 or m_perGroup > self.nDim:
            self.m_perGroup = self.nDim
        else:
            self.m_perGroup = m_perGroup

        self.nIter = nIter
        self.groups = int(np.floor((self.nDim - 1) / self.m_perGroup) + 1)
        self.momentum = momentum
        self.running_means = []
        self.running_projections = []

        self.sigmas = []
        self.set_Xs = []
        self.centereds = []
        self.whiten_matrixs = []
        self.eps = 1e-5

        groups = int(np.floor((self.nDim - 1) / self.m_perGroup) + 1)
        # allow nDim % m_perGroup != 0
        for i in range(0,  groups):
            if i < groups-1:
                self.running_means.append(tf.Variable(tf.zeros(self.m_perGroup), trainable=False))
                self.running_projections.append(tf.Variable(tf.eye(self.m_perGroup), trainable=False))
            else:
                self.running_means.append(tf.Variable(tf.zeros(self.nDim-(groups-1)*self.m_perGroup), trainable=False))
                self.running_projections.append(tf.Variable(tf.eye(self.nDim-(groups-1)*self.m_perGroup), trainable=False))

        if self.affine:
            self.weight = tf.Variable(tf.ones([self.nDim, ]))
            self.bias = tf.Variable(tf.zeros([self.nDim, ]))
            self.flag_inner_lr = False
            self.scale = 1


    def updateOutput_perGroup_train(self, data, groupId):
        nFeature = data.get_shape().as_list()[1]
        mean = tf.reduce_mean(data, 0)
        update_means = tf.assign(self.running_means[groupId],
                                 self.running_means[groupId] * (1 - self.momentum) + mean * self.momentum)

        centered = tf.expand_dims(data - mean, -1)
        # self.summaries.append(tf.summary.histogram('group{}_centered'.format(groupId), centered))

        sigma = tf.matmul(centered, tf.matrix_transpose(centered))
        sigma = tf.reduce_mean(sigma, 0)

        eig, rotation, _ = tf.svd(sigma)
        eig += self.eps
        eig = tf.pow(eig, -1/2)
        eig = tf.diag(eig)

        whitten_matrix =tf.matmul(rotation, eig)
        whitten_matrix = tf.matmul(whitten_matrix, tf.transpose(rotation))

        update_projections = tf.assign(self.running_projections[groupId],
                                 self.running_projections[groupId] * (1 - self.momentum) + whitten_matrix * self.momentum)

        self.centereds.append(centered)
        self.sigmas.append(sigma)
        self.whiten_matrixs.append(whitten_matrix)
        # self.set_Xs.append(set_X)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_means)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_projections)
        return tf.matmul(tf.squeeze(centered), whitten_matrix)

    def updateOutput_perGroup_test(self, data, groupId):
        # self.buffer = tf.tile(self.running_means[groupId], [nBatch])
        centered = data - self.running_means[groupId]
        return tf.matmul(centered, self.running_projections[groupId])

    def updateOutput(self, inputs, train):
        if len(self.input_shape) == 2:
            return self.updateOutput2D(inputs, train, self.affine)
        elif len(self.input_shape) == 4:
            return self.updateOutput4D(inputs, train)
        else:
            raise Exception('only 4D or 2D tensor supported, got {}D tensor instead'.format(len(self.input_shape)))

    def updateOutput2D(self, inputs, train, affine):
        outputs = []
        for i in range(self.groups):
            start_index = i * self.m_perGroup
            end_index = np.min(((i + 1) * self.m_perGroup, self.nDim))
            group_input = inputs[:, start_index:end_index]
            def updateOutputOnEvaluate():
                return self.updateOutput_perGroup_test(group_input, i)

            def updateOutputOnTrain():
                return self.updateOutput_perGroup_train(group_input, i)
            output = tf.cond(train, updateOutputOnTrain, updateOutputOnEvaluate)
            outputs.append(output)
        self.output = tf.concat(outputs, 1)

        # scale the output
        if affine:
            # multiply with gamma and add beta
            self.output = self.output * self.weight + self.bias
        return self.output

    def updateOutput4D(self, inputs, train):
        if self.data_format == 'channels_last':
            inputs_trans = tf.transpose(inputs, [3, 1, 2, 0])
        else:
            inputs_trans = tf.transpose(inputs, [1, 0, 2, 3])
        inputs_trans = tf.reshape(inputs_trans, [self.nDim, -1])
        inputs_trans = tf.transpose(inputs_trans, [1, 0])

        outputs = self.updateOutput2D(inputs_trans, train, self.affine)
        outputs = tf.transpose(outputs, [1, 0])
        if self.data_format == 'channels_last':
            outputs = tf.reshape(outputs, [self.nDim, self.iH, self.iW, -1])
            self.output = tf.transpose(outputs, [3, 1, 2, 0])
        else:
            outputs = tf.reshape(outputs, [self.nDim, -1, self.iH, self.iW])
            self.output = tf.transpose(outputs, [1, 0, 2, 3])

        return self.output



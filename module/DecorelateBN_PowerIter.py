"""
The Basic Decorelated Batch normalization version, in which:
(1) use ZCA to whitening the activation
(2) include train mode and test mode. in training mode, we train the module
"""
import numpy as np
import tensorflow as tf


def buildDBN(inputs, train, summary=[]):
    input_shape = inputs.get_shape().as_list()
    assert (len(input_shape) != 2 or len(input_shape) != 4), \
        'only 4D or 2D tensor supported, got {}D tensor instead'.format(len(input_shape))
    nDim = input_shape[1]
    DBN = DecorelateBN_PowerIter(nDim, train, summary)
    return DBN.updateOutput(inputs)


class DecorelateBN_PowerIter:
    def __init__(self, nDim, train, summary, m_perGroup=0, affine=False, nIter=5, momentum=0.1, debug=False, testMode_isRunning=True):
        self.affine = affine
        self.summaries = summary
        if m_perGroup:
            if m_perGroup == 0 or m_perGroup > nDim:
                self.m_perGroup = nDim
            else:
                self.m_perGroup = m_perGroup
        else:
            self.m_perGroup = nDim // 2

        self.nIter = nIter

        self.nDim = nDim  # the dimension of the input
        self.groups = int(np.floor((nDim - 1) / self.m_perGroup) + 1)
        self.momentum = momentum
        self.running_means = []
        self.running_projections = []

        self.sigmas = []
        self.set_Xs = []
        self.centereds = []
        self.whiten_matrixs = []

        groups = int(np.floor((nDim - 1) / self.m_perGroup) + 1)
        # allow nDim % m_perGroup != 0
        for i in range(0,  groups):
            if i < groups-1:
                self.running_means.append(tf.Variable(tf.zeros(self.m_perGroup), trainable=False))
                self.running_projections.append(tf.Variable(tf.eye(self.m_perGroup), trainable=False))
            else:
                self.running_means.append(tf.Variable(tf.zeros(nDim-(groups-1)*self.m_perGroup), trainable=False))
                self.running_projections.append(tf.Variable(tf.eye(nDim-(groups-1)*self.m_perGroup), trainable=False))

        if self.affine:
            print('---------------------------using scale-----------------')
            self.weight = tf.Variable(tf.truncated_normal([nDim, 1]))
            self.bias = tf.Variable(tf.zeros([nDim, 1]))
            self.flag_inner_lr = False
            self.scale = 1

        self.debug = debug
        # flag, whether is train mode. in train mode we do whitening
        # based on the mini-batch. in test mode, we use estimated parameters (running parameter)
        self.train = train
        # if this value set true, then use running parameter,
        # when do the training,  else false, use the previous parameters
        self.testMode_isRunning = testMode_isRunning
        self.count = 0
        self.printInterval = 1

    def updateOutput_perGroup_train(self, data, groupId):
        nFeature = data.get_shape().as_list()[1]
        mean = tf.reduce_mean(data, 0)
        update_means = tf.assign(self.running_means[groupId],
                                 self.running_means[groupId] * (1 - self.momentum) + mean * self.momentum)

        centered = tf.expand_dims(data - mean, -1)
        # self.summaries.append(tf.summary.histogram('group{}_centered'.format(groupId), centered))

        sigma = tf.matmul(centered, tf.matrix_transpose(centered))
        sigma = tf.reduce_mean(sigma, 0)
        # self.summaries.append(tf.summary.histogram('group{}_sigma'.format(groupId), sigma))

        trace = tf.trace(sigma)
        sigma_norm = sigma / trace

        set_X = []
        X = tf.eye(nFeature)
        for i in range(self.nIter):
            X = (3 * X - X * X * X * sigma_norm) / 2
            set_X.append(X)
        # self.summaries.append(tf.summary.histogram('group{}_X'.format(groupId), X))

        whitten_matrix = X / tf.sqrt(trace)
        # self.summaries.append(tf.summary.histogram('group{}_whitten_matrix'.format(groupId), whitten_matrix))

        update_projections = tf.assign(self.running_projections[groupId],
                                 self.running_projections[groupId] * (1 - self.momentum) + whitten_matrix * self.momentum)

        # self.summaries.append(tf.summary.histogram('group{}_running_mean'.format(groupId), self.running_means[groupId]))
        # self.summaries.append(tf.summary.histogram('group{}_running_projections'.format(groupId), self.running_projections[groupId]))

        if self.debug:
            pass

        self.centereds.append(centered)
        self.sigmas.append(sigma)
        self.whiten_matrixs.append(whitten_matrix)
        self.set_Xs.append(set_X)
        with tf.control_dependencies([update_means, update_projections]):
            return tf.matmul(tf.squeeze(centered), whitten_matrix)

    def updateOutput_perGroup_test(self, data, groupId):
        # self.buffer = tf.tile(self.running_means[groupId], [nBatch])
        centered = data - self.running_means[groupId]
        return tf.matmul(centered, self.running_projections[groupId])

    def updateOutput(self, inputs):
        outputs = []
        for i in range(self.groups):
            start_index = i * self.m_perGroup
            end_index = np.min(((i + 1) * self.m_perGroup, self.nDim))
            group_input = inputs[:, start_index:end_index]
            def updateOutputOnEvaluate():
                return self.updateOutput_perGroup_test(group_input, i)

            def updateOutputOnTrain():
                return self.updateOutput_perGroup_train(group_input, i)
            output = tf.cond(self.train, updateOutputOnTrain, updateOutputOnEvaluate)
            outputs.append(output)
        self.output = tf.concat(outputs, 1)

        # scale the output
        if self.affine:
            # multiply with gamma and add beta
            self.output = self.output * self.weight + self.bias
        return self.output

    def updateOutput4D(self, inputs):
        outputs = []
        for i in range(self.groups):
            start_index = i * self.m_perGroup
            end_index = np.min(((i + 1) * self.m_perGroup, self.nDim))
            group_input = inputs[:, start_index:end_index]
            def updateOutputOnEvaluate():
                return self.updateOutput_perGroup_test(group_input, i)

            def updateOutputOnTrain():
                return self.updateOutput_perGroup_train(group_input, i)
            output = tf.cond(self.train, updateOutputOnTrain, updateOutputOnEvaluate)
            outputs.append(output)
        self.output = tf.concat(outputs, 1)

        # scale the output
        if self.affine:
            # multiply with gamma and add beta
            self.output = self.output * self.weight + self.bias
        return self.output



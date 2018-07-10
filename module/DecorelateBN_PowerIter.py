"""
The Basic Decorelated Batch normalization version, in which:
(1) use ZCA to whitening the activation
(2) include train mode and test mode. in training mode, we train the module
"""
import numpy as np
import tensorflow as tf


def buildDBN(inputs, train, summary=[]):
    nDim = inputs.get_shape().as_list()[1]
    DBN = DecorelateBN_PowerIter(nDim, train, summary)
    return DBN.updateOutput(inputs)


class DecorelateBN_PowerIter:
    def __init__(self, nDim, train, summary, m_perGroup=None, affine=False, nIter=5, momentum=0.1, debug=False, testMode_isRunning=True):
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

        print('m_perGroup:', self.m_perGroup, '----nIter:', self.nIter)

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
                self.running_means.append(tf.zeros(self.m_perGroup))
                self.running_projections.append(tf.eye(self.m_perGroup))
            else:
                self.running_means.append(tf.zeros(nDim-(groups-1)*self.m_perGroup))
                self.running_projections.append(tf.eye(nDim-(groups-1)*self.m_perGroup))

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
        self.running_means[groupId] = self.running_means[groupId] * (1 - self.momentum) + self.momentum * mean

        centered = tf.expand_dims(data - mean, -1)
        self.summaries.append(tf.summary.histogram('group{} centered'.format(groupId), centered))

        sigma = tf.matmul(centered, tf.matrix_transpose(centered))
        sigma = tf.reduce_mean(sigma, 0)
        self.summaries.append(tf.summary.histogram('group{} sigma'.format(groupId), sigma))

        trace = tf.trace(sigma)
        sigma_norm = sigma / trace

        set_X = []
        X = tf.eye(nFeature)
        for i in range(self.nIter):
            X = (3 * X - X * X * X * sigma_norm) / 2
            set_X.append(X)
        self.summaries.append(tf.summary.histogram('group{} X'.format(groupId), X))

        whitten_matrix = X / tf.sqrt(trace)
        self.summaries.append(tf.summary.histogram('group{} whitten_matrix'.format(groupId), whitten_matrix))
        self.running_projections[groupId] = self.running_projections[groupId] * (
                    1 - self.momentum) + self.momentum * whitten_matrix

        self.summaries.append(tf.summary.histogram('group{} running mean'.format(groupId), self.running_means[groupId]))
        self.summaries.append(tf.summary.histogram('group{} running projections'.format(groupId), self.running_projections[groupId]))

        if self.debug:
            pass

        self.centereds.append(centered)
        self.sigmas.append(sigma)
        self.whiten_matrixs.append(whitten_matrix)
        self.set_Xs.append(set_X)

        return tf.matmul(tf.squeeze(centered), whitten_matrix)

    def updateOutput_perGroup_test(self, data, groupId):
        # self.buffer = tf.tile(self.running_means[groupId], [nBatch])
        centered = data - self.running_means[groupId]
        return tf.matmul(centered, self.running_projections[groupId])

    def updateOutput(self, inputs):
        outputs_train = []
        outputs_test = []
        self.output = tf.zeros_like(inputs)
        def updateOutputOnEvaluate():
            for i in range(self.groups):
                start_index = i * self.m_perGroup
                end_index = np.min(((i+1) * self.m_perGroup, self.nDim))
                output = self.updateOutput_perGroup_test(inputs[:, start_index:end_index], i)
                outputs_test.append(output)
                # self.output[:, start_index:end_index] = self.updateOutput_perGroup_test(inputs[:, start_index:end_index], i)
            result = tf.concat(outputs_test, 1)
            return result

        def updateOutputOnTrain():
            for i in range(self.groups):
                start_index = i * self.m_perGroup
                end_index = np.min(((i+1) * self.m_perGroup, self.nDim))
                output = self.updateOutput_perGroup_train(inputs[:, start_index:end_index], i)
                outputs_train.append(output)
                # self.output[:, start_index:end_index] = self.updateOutput_perGroup_train(inputs[:, start_index:end_index], i)
            result = tf.concat(outputs_train, 1)
            return result

        self.output = tf.cond(self.train, updateOutputOnTrain, updateOutputOnEvaluate)

        # scale the output
        if self.affine:
            # multiply with gamma and add beta
            self.output = self.output * self.weight + self.bias
        return self.output



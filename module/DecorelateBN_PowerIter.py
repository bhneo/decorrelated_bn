"""
The Basic Decorelated Batch normalization version, in which:
(1) use ZCA to whitening the activation
(2) include train mode and test mode. in training mode, we train the module
"""
import numpy as np
import tensorflow as tf


def buildDBN(inputs, train):
    nDim = inputs.get_shape()[1]
    DBN = DecorelateBN_PowerIter(nDim, train)
    return DBN.updateOutput(inputs)


class DecorelateBN_PowerIter:
    def __init__(self, nDim, train, m_perGroup=None, affine=False, nIter=5, momentum=0.1, debug=False, testMode_isRunning=True):
        self.affine = affine
        if m_perGroup:
            if m_perGroup == 0 or m_perGroup > nDim:
                self.m_perGroup = nDim
            else:
                self.m_perGroup = m_perGroup
        else:
            self.m_perGroup = nDim / 2

        if nIter:
            self.nIter = nIter
        else:
            self.nIter = 5

        print('m_perGroup:', self.m_perGroup, '----nIter:', self.nIter)

        self.nDim = nDim  # the dimension of the input
        self.groups = np.floor((nDim - 1) / self.m_perGroup) + 1
        self.momentum = momentum
        self.running_means = []
        self.running_projections = []

        self.summaries = []

        self.sigmas = []
        self.set_Xs = []
        self.centereds = []
        self.whiten_matrixs = []

        groups = np.floor((nDim - 1) / self.m_perGroup) + 1
        # allow nDim % m_perGroup != 0
        for i in range(start=1, stop=groups+1):
            if i < groups:
                self.running_means.append(tf.zeros(m_perGroup))
                self.running_projections.append(tf.eye(m_perGroup))
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
        nFeature = data.get_shape()[1]
        mean = tf.reduce_mean(data, 0)
        self.running_means[groupId] = self.running_means[groupId] * (1 - self.momentum) + self.momentum * mean

        centered = data - mean

        sigma = tf.reduce_mean(tf.matmul(tf.matrix_transpose(centered), centered), 0)

        trace = tf.trace(sigma)
        sigma_norm = sigma / trace

        set_X = []
        X = tf.eye(nFeature)
        for i in range(self.nIter):
            X = (3 * X - X * X * X * sigma_norm) / 2
            set_X.append(X)

        whitten_matrix = X / tf.sqrt(trace)
        self.running_projections[groupId] = self.running_projections[groupId] * (
                    1 - self.momentum) + self.momentum * whitten_matrix

        if self.debug:
            pass

        self.centereds.append(centered)
        self.sigmas.append(sigma)
        self.whiten_matrixs.append(whitten_matrix)
        self.set_Xs.append(set_X)

        return centered, whitten_matrix

    def updateOutput_perGroup_test(self, nBatch, data, groupId):
        self.buffer = tf.tile(self.running_means[groupId], [nBatch])
        self.buffer_1 = data - self.buffer
        return self.buffer_1, self.running_projections[groupId]

    def updateOutput(self, inputs):
        """
        self.output = self.output or input.new()
        self.output: resizeAs(input)
    
        self.gradInput = self.gradInput or input.new()
        self.gradInput: resizeAs(input)
    
        self.normalized = self.normalized or input.new()
        #- -used for the affine transformation to calculate the gradient
        self.normalized: resizeAs(input)
        # buffers that are reused
        self.buffer = self.buffer or input.new()
        self.buffer_1 = self.buffer_1 or input.new()
        self.buffer_2 = self.buffer_2 or input.new()
        """
        def updateOutputOnEvaluate():
            for i in range(self.groups):
                start_index = i * self.m_perGroup
                end_index = np.min(((i+1) * self.m_perGroup, self.nDim))
                self.output[:, start_index:end_index] = self.updateOutput_perGroup_test(inputs[:, start_index:end_index], i)

        def updateOutputOnTrain():
            for i in range(self.groups):
                start_index = i * self.m_perGroup
                end_index = np.min(((i+1) * self.m_perGroup, self.nDim))
                self.output[:, start_index:end_index] = self.updateOutput_perGroup_train(inputs[:, start_index:end_index], i)

        tf.cond(self.train, updateOutputOnTrain, updateOutputOnEvaluate)

        # scale the output
        if self.affine:
            # multiply with gamma and add beta
            self.output = self.output * self.weight + self.bias
        return self.output



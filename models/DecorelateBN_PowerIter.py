"""
The Basic Decorelated Batch normalization version, in which:
(1) use ZCA to whitening the activation
(2) include train mode and test mode. in training mode, we train the module
"""
import numpy as np
import tensorflow as tf


class DecorelateBN_PowerIter:
    def __init__(self, nDim, m_perGroup, affine, nIter, momentum):
        if affine:
            assert type(affine) == bool
            self.affine = affine
        else:
            self.affine = False

        if m_perGroup:
            self.m_perGroup = m_perGroup == 0 and nDim or m_perGroup > nDim and nDim or m_perGroup
        else:
            self.m_perGroup = nDim / 2

        if nIter:
            self.nIter = nIter
        else:
            self.nIter = 5

        print('m_perGroup:', self.m_perGroup, '----nIter:', self.nIter)

        self.nDim = nDim  # the dimension of the input
        self.momentum = momentum or 0.1
        self.running_means = []
        self.running_projections = []
        self.centereds = []
        self.sigmas = []
        self.whiten_matrixs = []
        self.set_Xs = []

        groups = np.floor((nDim - 1) / self.m_perGroup) + 1
        # allow nDim % m_perGroup != 0
        for i in range(start=1, stop=groups+1):
            if i < groups:
                self.r_mean = np.zeros(m_perGroup)
                self.r_projection = np.eye(m_perGroup)
            else:
                self.r_mean = np.zeros(nDim-(groups-1)*self.m_perGroup)
                self.r_projection = np.eye(nDim-(groups-1)*self.m_perGroup)

        if self.affine:
            print('---------------------------using scale-----------------')
            self.weight = tf.Variable(tf.ones([nDim, 1]))
            self.bias = tf.Variable(tf.zeros([nDim, 1]))
            self.gradWeight = tf.Variable(tf.truncated_normal([nDim, 1]))
            self.gradBias = tf.Variable(tf.truncated_normal([nDim, 1]))
            self.flag_inner_lr = False
            self.scale = 1

        # flag, whether is train mode. in train mode we do whitening
        # based on the mini-batch. in test mode, we use estimated parameters (running parameter)
        self.debug = False
        self.train = True
        # if this value set true, then use running parameter,
        # when do the training,  else false, use the previous parameters
        self.count = 0
        self.printInterval = 1

    def updateOutput(self, inputs):
        def updateOutput_perGroup_train(nBatch, data, groupId):
            nFeature = data.get_shape()[1]
            mean = tf.reduce_mean(data, 0)
            self.running_means[groupId] = self.running_means[groupId] * (1 - self.momentum) + self.momentum * mean

            centered = data - mean

            sigma = tf.reduce_mean(tf.matmul(tf.matrix_transpose(centered), centered), 0)

            trace = tf.trace(sigma)
            sigma_norm = sigma/trace

            set_X = []
            X = tf.eye(nFeature)
            for i in range(self.nIter):
                X = (3*X-X*X*X*sigma_norm) / 2
                set_X.append(X)

            whitten_matrix = X/tf.sqrt(trace)
            self.running_projections[groupId] = self.running_projections[groupId] * (1 - self.momentum) + self.momentum * whitten_matrix

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



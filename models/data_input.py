import os
import tensorflow as tf
import numpy as np


def load_mnist(batch_size, is_training=True, spatial=False):
    path = os.path.join('data', 'mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        if spatial:
            trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)
        else:
            trainX = loaded[16:].reshape((60000, 784)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size

        return trX, trY, valX, valY, 10, num_tr_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        if spatial:
            teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)
        else:
            teX = loaded[16:].reshape((10000, 784)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_fashion_mnist(batch_size, is_training=True, spatial=False):
    path = os.path.join('data', 'fashion-mnist')
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        if spatial:
            trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)
        else:
            trainX = loaded[16:].reshape((60000, 784)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:55000] / 255.
        trY = trainY[:55000]

        valX = trainX[55000:, ] / 255.
        valY = trainY[55000:]

        num_tr_batch = 55000 // batch_size

        return trX, trY, valX, valY, 10, num_tr_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        if spatial:
            teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)
        else:
            teX = loaded[16:].reshape((10000, 784)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch


def load_data(dataset, batch_size, is_training=True, spatial=False):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training, spatial)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training, spatial)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def create_train_set(dataset, handle, spatial=False, batch_size=128, n_repeat=-1):
    tr_x, tr_y, val_x, val_y, num_label, num_batch = load_data(dataset, batch_size, is_training=True, spatial=spatial)

    tr_data_set = tf.data.Dataset.from_tensor_slices((tr_x, tr_y)).repeat(n_repeat).batch(batch_size)
    val_data_set = tf.data.Dataset.from_tensor_slices((val_x, val_y)).repeat(n_repeat).batch(batch_size)

    feed_iterator = tf.data.Iterator.from_string_handle(handle, tr_data_set.output_types,
                                                        tr_data_set.output_shapes)
    X, y = feed_iterator.get_next()
    # 创建不同的iterator
    train_iterator = tr_data_set.make_one_shot_iterator()
    val_iterator = val_data_set.make_one_shot_iterator()

    return X, y, train_iterator, val_iterator, num_label, num_batch


import data_input
import model_builder
import os

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from config import cfg
from tensorflow import keras
from layers import DecorrelatedBN, IterativeNormalization, DecorelateBNPowerIter2
from tensorflow.python.keras import layers
from tensorflow.python.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.backend import clear_session


def build_vgg(method, filters, repeats, out_num, weight_decay, height=32, width=32, depth=3, m_per_group=16, dbn_affine=True):
    regularizer = keras.regularizers.l2(weight_decay) if weight_decay != 0 else None
    input_image = tf.placeholder(dtype=tf.float32, shape=[None, height, width, depth], name='input_image')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    x = input_image
    for repeat in repeats:
        for _ in range(repeat):
            x = layers.Conv2D(filters=filters, kernel_size=3, padding='same',
                              kernel_regularizer=regularizer)(x)
            if method == 'bn':
                x = layers.BatchNormalization()(x, training=is_training)
            elif method == 'dbn':
                x = DecorrelatedBN(m_per_group=m_per_group, affine=dbn_affine)(x, training=is_training)
                # x = DecorelateBNPowerIter2.buildDBN(x, train=is_training, per_group=m_per_group)
            elif method == 'iter_norm':
                x = IterativeNormalization(m_per_group=m_per_group, affine=dbn_affine)(x, training=is_training)
            x = layers.ReLU()(x)

        x = layers.MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
        if filters < 512:
            filters *= 2

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x, training=is_training)
    x = layers.Dense(512, kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x, training=is_training)
    x = layers.ReLU()(x)
    out = layers.Dense(out_num, kernel_regularizer=regularizer)(x)

    return input_image, is_training, out


def train(dataset, method, optimizer, tensorboard=False, learning_rate_scheduler=None):
    train_images, train_labels, test_images, test_labels = data_input.load_data_on_memory(dataset)

    input_image, is_training, out = build_vgg(method=method, filters=64, weight_decay=0.0005, height=32,
                              width=32, depth=3, m_per_group=16, dbn_affine=True, out_num=10, repeats=[1,1,2,2,2])
    input_label = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='input_label')
    loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(input_label, out))

    train_batch_num = len(train_images) // cfg.batch_size
    test_batch_num = len(test_images) // cfg.batch_size

    optimizer = tf.train.AdamOptimizer(0.1)
    opt = optimizer.minimize(loss)
    acc = tf.reduce_mean(keras.metrics.sparse_categorical_accuracy(input_label, out))

    # Create the metrics
    loss_metric = tf.keras.metrics.Mean(name='train_loss')
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    history = {'acc':[], 'val_acc':[], 'loss':[], 'val_loss':[]}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(cfg.epochs):
            # Reset the metrics
            loss_metric.reset_states()
            accuracy_metric.reset_states()
            for i in range(train_batch_num):
                start = i*cfg.batch_size
                end = (i+1)*cfg.batch_size
                image_batch = train_images[start:end]
                label_batch = train_labels[start:end]
                _, outputs, _loss, _acc = sess.run([opt, out, loss, acc], feed_dict={input_image: image_batch,
                                                                  input_label: label_batch,
                                                                  is_training: True})

                # Update the metrics
                loss_metric.update_state(_loss)
                accuracy_metric.update_state(label_batch, outputs)

                print('loss:{}, acc:{}'.format(_loss, _acc))

                # Get the metric results
            mean_loss = loss_metric.result()
            mean_accuracy = accuracy_metric.result()

            print('Epoch: ', e)
            print('  loss:     {}'.format(mean_loss))
            print('  accuracy: {}'.format(mean_accuracy))

    return history


def get_optimizer(method, lr, momentum=0.9):
    if method == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr, momentum=momentum)
    elif method == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr)
    else:
        optimizer = tf.keras.optimizers.Adam(lr)
    return optimizer


def main(_):
    if cfg.strategy == 'debug':
        pass
    elif cfg.strategy == 'vggA_base':
        methods = [ 'iter_norm', 'dbn','bn']
        lr = 0.1
        cfg.epochs = 50
        cfg.batch_size = 256
        cfg.augment = True
        if os.path.exists(cfg.result + '/vggA_base.csv'):
            os.remove(cfg.result + '/vggA_base.csv')
        df = pd.DataFrame()
        fig_acc = plt.figure(num='fig_acc')
        fig_loss = plt.figure(num='fig_loss')
        legends = []
        for method in methods:
            tf.reset_default_graph()
            print('method:{}'.format(method))
            print('lr:{}'.format(lr))
            optimizer = get_optimizer('sgd', lr, momentum=0.9)

            def lr_scheduler(epoch, lr):
                decay_rate = 0.5
                decay_step = 20
                if epoch % decay_step == 0 and epoch:
                    return lr * decay_rate
                return lr

            plot_name = '_'.join([method, str(lr)])
            # history = train('fashion_mnist', model, optimizer, learning_rate_scheduler=LearningRateScheduler(lr_scheduler))
            history = train('cifar10', method, tf.train.AdamOptimizer(lr))



if __name__ == "__main__":
    tf.app.run()

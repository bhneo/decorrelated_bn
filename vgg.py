import os
import sys

sys.path.append(os.getcwd())

import tensorflow as tf
from tensorflow import keras
from common import normalization, utils, train
import data_input
import config


WEIGHT_DECAY = 1e-4

kernel_regularizer = keras.regularizers.l2(WEIGHT_DECAY)
kernel_initializer = keras.initializers.he_normal()
BASE_NAME = 'vgg'

arch = {
    'A': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512],
    'B': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


def build_model_name(params):
    model_name = BASE_NAME
    model_name += '_{}'.format(params.model.type)
    model_name += '_bs{}'.format(params.training.batch_size)
    model_name += '_lr{}'.format(params.training.lr)
    model_name += '_{}'.format(params.normalize.method)
    if params.normalize.m != 0:
        model_name += '_m{}'.format(params.normalize.m)
    if params.normalize.method == 'iter_norm':
        model_name += '_iter{}'.format(params.normalize.iter)
    if params.normalize.affine:
        model_name += '_affine'
    model_name += '_idx{}'.format(str(params.training.idx))
    return model_name


def build_model(shape, num_out, params):
    inputs = keras.Input(shape=shape)
    model_name = build_model_name(params)

    probs, tensor_log = build(inputs, num_out,
                              arch[params.model.type],
                              params.normalize.method,
                              params.normalize.m,
                              params.normalize.iter)
    model = keras.Model(inputs=inputs, outputs=probs, name=model_name)
    log_model = keras.Model(inputs=inputs, outputs=tensor_log.get_outputs(), name=model_name + '_log')
    tensor_log.set_model(log_model)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.SGD(lr=params.training.lr, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[])
    model.summary()
    lr_scheduler = keras.callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=1)
    lr_scheduler.set_model(model)
    callbacks = [lr_scheduler]
    model.callbacks = callbacks
    return model, tensor_log


def build(inputs, num_out, arch, method, m, iter):
    log = utils.TensorLog()
    feature = inputs
    for i, layer in enumerate(arch):
        if layer == 'M':
            feature = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(feature)
        else:
            feature = keras.layers.Conv2D(filters=layer, kernel_size=3,
                                          strides=1, padding='same', use_bias=False,
                                          kernel_initializer=kernel_initializer,
                                          kernel_regularizer=kernel_regularizer)(feature)
            if method == 'bn':
                feature = keras.layers.BatchNormalization()(feature)
            elif method == 'zca':
                feature = normalization.DecorelationNormalization(m_per_group=m,
                                                                  decomposition='zca_wm')(feature)
            elif method == 'iter_norm':
                feature = normalization.DecorelationNormalization(m_per_group=m,
                                                                  decomposition='iter_norm_wm',
                                                                  iter_num=iter)(feature)
            log.add_hist('bn{}'.format(i+1), feature)
            feature = keras.layers.ReLU()(feature)

    feature = keras.layers.AveragePooling2D(pool_size=(2, 2))(feature)
    feature = keras.layers.Flatten()(feature)
    output = keras.layers.Dense(num_out)(feature)

    return output, log


def build_parse(flip=False, crop=False, is_train=False):
    height, width, channel = 32, 32, 3
    mean = [0.491, 0.482, 0.447]
    std = [0.247, 0.243, 0.262]

    def parse(image, label):
        image = tf.cast(image, tf.float32)
        if is_train:
            if crop:
                image = tf.image.resize_with_crop_or_pad(image, height+8, width+8)
                image = tf.image.random_crop(image, [height, width, channel])
            if flip:
                image = tf.image.random_flip_left_right(image)
        image = tf.divide(image, 255.)
        image = (image - mean) / std
        return image, label
    return parse


def lr_schedule(epoch, lr):
    if epoch in [60, 120]:
        lr /= 5
    return lr


def main():
    args, params = config.parse_args()
    train_set, test_set, info = data_input.build_dataset('cifar10',
                                                         parser_train=build_parse(flip=params.dataset.flip,
                                                                                  crop=params.dataset.crop,
                                                                                  is_train=True),
                                                         parser_test=build_parse(is_train=False),
                                                         batch_size=params.training.batch_size)
    model, tensor_log = build_model(shape=info.features['image'].shape,
                                    num_out=info.features['label'].num_classes,
                                    params=params)

    trainer = train.Trainer(model, params, info, tensor_log)
    if args.train:
        trainer.fit(train_set, test_set)
    else:
        trainer.evaluate(test_set)


def test_build_dataset():
    train_set, test_set, info = data_input.build_dataset('cifar10',
                                                         parser_train=build_parse(flip=True,
                                                                                  crop=True,
                                                                                  is_train=True),
                                                         parser_test=build_parse(is_train=False),
                                                         batch_size=128)
    for image, label in train_set:
        data_input.out_image(image, label)
        break

    for image, label in test_set:
        data_input.out_image(image, label)
        break


def test_build():
    inputs = keras.layers.Input([32, 32, 3])
    # inputs = tf.random.normal([128, 32, 32, 3])
    outputs, _ = build(inputs, 10, arch['E'], 'dbn', 16, 5)
    model = keras.Model(inputs=inputs, outputs=outputs, name='')
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[])
    model.summary()


if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow import keras

from common import blocks, layers, utils
from config import config as cfg

WEIGHT_DECAY = 5e-4

kernel_regularizer = keras.regularizers.l2(WEIGHT_DECAY)
kernel_initializer = keras.initializers.he_normal()

model_name = '_'.join([cfg.model.name, str(cfg.model.layer_num)])
model_name += '_batch{}'.format(str(cfg.training.batch_size))
model_name += '_lr{}'.format(str(cfg.training.lr))
if cfg.normalize.type == 'dbn' or cfg.normalize.type == 'iter_norm':
    model_name = '_'.join([model_name, cfg.normalize.type])
    model_name = '_'.join([model_name, str(cfg.normalize.m)])
    if cfg.normalize.affine:
        model_name = '_'.join([model_name, 'affine'])
    if cfg.normalize.type == 'iter_norm':
        model_name += '_iter{}'.format(str(cfg.normalize.iter))

model_name += '_trial{}'.format(str(cfg.training.idx))


def build_model(shape, num_out):
    if cfg.model.layer_num == 11:  # A
        repetitions = [1, 1, 2, 2, 2]
    elif cfg.model.layer_num == 13:  # B
        repetitions = [2, 2, 2, 2, 2]
    elif cfg.model.layer_num == 16:  # D
        repetitions = [2, 2, 3, 3, 3]
    elif cfg.model.layer_num == 19:  # E
        repetitions = [2, 2, 4, 4, 4]

    inputs = keras.Input(shape=shape)
    prob, tensor_log = build(inputs, num_out, blocks.vgg_block, repetitions, cfg.normalize.type, cfg.normalize.m, cfg.normalize.affine)
    model = keras.Model(inputs=inputs, outputs=prob, name=model_name)
    log_model = keras.Model(inputs=inputs, outputs=tensor_log.get_outputs(), name=model_name + '_log')
    tensor_log.set_model(log_model)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=cfg.training.lr, momentum=cfg.training.momentum),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[])
    model.summary()

    lr_scheduler = keras.callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=1)
    lr_scheduler.set_model(model)
    callbacks = [lr_scheduler]
    model.callbacks = callbacks
    return model, tensor_log


def build_output(out_num, x):
    x = keras.layers.Flatten()(x)
    # x = keras.layers.Dropout(0.5)(x)
    # x = keras.layers.Dense(512, kernel_regularizer=kernel_regularizer)(x)
    # x = keras.layers.BatchNormalization()(x)
    # x = keras.layers.ReLU()(x)
    out = keras.layers.Dense(out_num)(x)
    return out


def build(inputs, num_out, block_fn, repetitions, bn_type, m, affine):
    log = utils.TensorLog()
    backbone = blocks.build_vgg_backbone(inputs, repetitions, block_fn, bn=bn_type, m=m, affine=affine, start_filters=64, weight_decay=WEIGHT_DECAY)
    log.add_hist('backbone', backbone)

    prob = build_output(num_out, backbone)
    return prob, log


def lr_schedule(epoch, lr):
    if epoch in [20, 40, 60, 80]:
        lr /= 2
    return lr


def test_build():
    inputs = tf.random.normal([128, 32, 32, 3])
    outputs = build(inputs, 10, blocks.vgg_block, [1, 1, 2, 2, 2], 'dbn', 16, True)

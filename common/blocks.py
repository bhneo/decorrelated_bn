from tensorflow.python.keras import regularizers, initializers
from tensorflow import keras
from common import layers


WEIGHT_DECAY = 1e-4


def bn_relu(inputs, bn=keras.layers.BatchNormalization()):
    x = bn(inputs)
    x = keras.layers.ReLU()(x)
    return x


def fc_bn(inputs, num_output, is_relu=True, weight_decay=1e-4):
    out = keras.layers.Dense(num_output,
                       kernel_initializer=initializers.glorot_normal(),
                       kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    out = keras.layers.BatchNormalization()(out)
    if is_relu is True:
        return keras.layers.ReLU()(out)
    else:
        return out


def conv_bn_relu(inputs, filters=64, kernel_size=(7, 7), strides=(2, 2),
                 bn=keras.layers.BatchNormalization(), use_bias=True,
                 kernel_initializer=initializers.he_normal(),
                 kernel_regularizer=regularizers.l2(WEIGHT_DECAY)):
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                            strides=strides, padding='same', use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer)(inputs)
    return bn_relu(x, bn)


def bn_relu_conv(inputs, filters, kernel_size, strides=(1, 1), padding='same',
                 bn=keras.layers.BatchNormalization(), use_bias=True,
                 kernel_initializer=initializers.he_normal(),
                 kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                 name=None):
    x = bn_relu(inputs, bn)
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                            strides=strides, padding=padding, use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            name='res'+name)(x)
    return x


def shortcut_v2(inputs, residual,
                use_bias=True,
                kernel_initializer=initializers.he_normal(),
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                name=None):
    input_shape = inputs.get_shape().as_list()
    residual_shape = residual.get_shape().as_list()
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    stride_height = int(round(input_shape[1] / residual_shape[1]))
    equal_channels = input_shape[3] == residual_shape[3]

    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        inputs = keras.layers.Conv2D(filters=residual_shape[3], kernel_size=(1, 1),
                                     strides=(stride_width, stride_height), use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     name='res'+name)(inputs)
    return keras.layers.add([inputs, residual])


def basic_block_v2(inputs, filters,
                   stage, block,
                   use_bias=True,
                   init_strides=(1, 1), is_first_block_of_first_layer=False,
                   kernel_initializer=initializers.he_normal(),
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY)):
    if isinstance(filters, int):
        filters1, filters2 = filters, filters
    else:
        filters1, filters2 = filters
    base_name = str(stage) + block + '_branch'
    if is_first_block_of_first_layer:
        x = keras.layers.Conv2D(filters=filters1, kernel_size=(3, 3), use_bias=use_bias,
                                strides=init_strides, padding='same',
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer,
                                name=base_name+'2a')(inputs)
    else:
        x = bn_relu_conv(inputs, filters=filters1, kernel_size=(3, 3), use_bias=use_bias,
                         strides=init_strides,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         name=base_name+'2a')

    residual = bn_relu_conv(x, filters=filters2, kernel_size=(3, 3), strides=(1, 1),
                            use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            name=base_name+'2b')
    return shortcut_v2(inputs, residual,
                       use_bias=use_bias,
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=kernel_regularizer,
                       name=base_name + '1')


def bottleneck_v2(inputs, filters, init_strides=(1, 1),
                  use_bias=True,
                  is_first_block_of_first_layer=False,
                  kernel_initializer=initializers.he_normal(),
                  kernel_regularizer=regularizers.l2(WEIGHT_DECAY)):
    if is_first_block_of_first_layer:
        x = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1),
                                strides=init_strides, use_bias=use_bias, padding='same',
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer)(inputs)
    else:
        x = bn_relu_conv(inputs, filters=filters, kernel_size=(1, 1),
                         strides=init_strides, use_bias=use_bias)

    x = bn_relu_conv(x, filters=filters, kernel_size=(3, 3), use_bias=use_bias)
    x = bn_relu_conv(x, filters=filters * 4, kernel_size=(1, 1), use_bias=use_bias)
    return shortcut_v2(inputs, x, use_bias=use_bias)


def residual_block(inputs, block_function, filters, repetitions, stage, use_bias=True,
                   kernel_initializer=initializers.he_normal(),
                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                   ):
    x = inputs
    for i in range(repetitions):
        init_strides = (1, 1)
        if i == 0 and stage != 2:
            init_strides = (2, 2)
        x = block_function(x, filters=filters,
                           stage=stage, block='block_%d' % (i + 1), use_bias=use_bias,
                           init_strides=init_strides,
                           is_first_block_of_first_layer=(stage == 2 and i == 0),
                           kernel_initializer=kernel_initializer,
                           kernel_regularizer=kernel_regularizer,
                           )
    return x


def build_resnet_backbone(inputs, layer_num, repetitions=None, start_filters=16,
                          arch='cifar', use_bias=True,
                          kernel_initializer=initializers.he_normal(),
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                          ):
    if arch == 'cifar':
        if repetitions is None:
            if layer_num == 18:
                repetitions = [2, 2, 2, 2]
            elif layer_num == 20:
                repetitions = [3, 3, 3]
            elif layer_num == 32:
                repetitions = [5, 5, 5]
            elif layer_num == 44:
                repetitions = [7, 7, 7]
            elif layer_num == 56:
                repetitions = [9, 9, 9]
            elif layer_num == 110:
                repetitions = [18, 18, 18]
            else:
                repetitions = [3, 3, 3]

    conv1 = conv_bn_relu(inputs, filters=start_filters, kernel_size=(3, 3), strides=(1, 1))

    block = conv1
    filters = start_filters
    for i, r in enumerate(repetitions):
        block = residual_block(block, block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))
        filters *= 2
    return block


def vgg_block(inputs, layer_num, filters,
              kernel_size=(3, 3), strides=(1, 1),
              pool_size=(2, 2), pool_strides=(2, 2),
              bn_type='dbn', m=16, affine=True, iter=5,
              weight_decay=1e-4):
    conv = inputs
    for i in range(layer_num):
        if bn_type == 'dbn':
            bn = layers.DecorrelatedBN(m_per_group=m, affine=affine)
        elif bn_type == 'iter_norm':
            bn = layers.IterativeNormalization(m_per_group=m, affine=affine, iter_num=iter)
        else:
            bn = keras.layers.BatchNormalization()
        conv = conv_bn_relu(conv, filters=filters, kernel_size=kernel_size, strides=strides,
                            bn=bn,
                            kernel_initializer=initializers.he_normal(),
                            kernel_regularizer=regularizers.l2(weight_decay))
    pose = keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_strides, padding='same')(conv)
    return pose


def build_vgg_backbone(inputs, repetitions, block_fn, bn, m=16, affine=True, iter=5, start_filters=16, weight_decay=1e-4):
    block = inputs
    filters = start_filters
    for i, layer_num in enumerate(repetitions):
        block = block_fn(block, layer_num, filters, bn_type=bn, m=m, affine=affine, iter=iter, weight_decay=weight_decay)
        if filters < 512:
            filters *= 2
    return block


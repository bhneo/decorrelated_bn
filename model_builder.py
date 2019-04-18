from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Layer
from layers import DecorrelatedBN, IterativeNormalization
from config import cfg

import  tensorflow as tf

def build_mlp(method, hidden_layers, out_num, weight_decay, height=32, width=32, depth=3, m_per_group=0, dbn_affine=True):
    regularizer = keras.regularizers.l2(weight_decay) if weight_decay != 0 else None
    input_image = layers.Input(shape=(height, width, depth))
    x = layers.Reshape(target_shape=[height*width*depth])(input_image)
    x = layers.Dense(hidden_layers[0], kernel_regularizer=regularizer)(x)
    for hidden in hidden_layers:
        if method == 'plain':
            pass
        elif method == 'bn':
            x = layers.BatchNormalization()(x)
        elif method == 'dbn':
            x = DecorrelatedBN(m_per_group=m_per_group, affine=dbn_affine)(x)
        elif method == 'iter_norm':
            x = IterativeNormalization(m_per_group=m_per_group, affine=dbn_affine)(x)
        x = layers.Activation(cfg.activation)(x)
        x = layers.Dense(hidden, kernel_regularizer=regularizer)(x)

    out = layers.Dense(out_num, activation='softmax', kernel_regularizer=regularizer)(x)

    model = keras.Model(inputs=input_image, outputs=out)
    return model


def build_vgg(method, filters, repeats, out_num, weight_decay, height=32, width=32, depth=3, m_per_group=16, dbn_affine=True):
    regularizer = keras.regularizers.l2(weight_decay) if weight_decay != 0 else None
    input_image = layers.Input(shape=(height, width, depth))
    x = input_image
    for repeat in repeats:
        for _ in range(repeat):
            x = layers.Conv2D(filters=filters, kernel_size=3, padding='same',
                              kernel_regularizer=regularizer)(x)
            if method == 'bn':
                x = layers.BatchNormalization()(x)
            elif method == 'dbn':
                x = DecorrelatedBN(m_per_group=m_per_group, affine=dbn_affine)(x)
            elif method == 'iter_norm':
                x = IterativeNormalization(m_per_group=m_per_group, affine=dbn_affine)(x)
            x = layers.ReLU()(x)

        x = layers.MaxPool2D(padding='same')(x)

        filters *= 2

    x = layers.Flatten()(x)
    x = layers.Dense(512, kernel_regularizer=regularizer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    out = layers.Dense(out_num, activation='softmax', kernel_regularizer=regularizer)(x)

    model = keras.Model(inputs=input_image, outputs=out)
    return model



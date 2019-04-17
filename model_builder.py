from tensorflow import keras
from tensorflow.python.keras import layers
from layers import DecorrelatedBN, DecorrelatedBNPowerIter
from config import cfg


def build_mlp(method, hiddens, out_num, weight_decay, height=32, width=32, depth=3, m_per_group=0, dbn_affine=True):
    regularizer = keras.regularizers.l2(weight_decay) if weight_decay != 0 else None
    input_image = layers.Input(shape=(height, width, depth))
    x = layers.Reshape(target_shape=[height*width*depth])(input_image)
    for hidden in hiddens:
        if method == 'plain':
            pass
        elif method == 'bn':
            x = layers.BatchNormalization()(x)
        elif method == 'dbn':
            x = DecorrelatedBN(m_per_group=m_per_group, affine=dbn_affine)(x)
        elif method == 'dbn_iter':
            x = DecorrelatedBNPowerIter(m_per_group=m_per_group, affine=dbn_affine)(x)
        x = layers.Activation(cfg.activation)(x)
        x = layers.Dense(hidden, kernel_regularizer=regularizer)(x)

    out = layers.Dense(out_num, activation='softmax', kernel_regularizer=regularizer)(x)

    model = keras.Model(inputs=input_image, outputs=out)
    return model


def build_cnn():
    layers = []

    model = keras.Sequential(layers)
    return model



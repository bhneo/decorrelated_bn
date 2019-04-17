from config import cfg
import data_input
import model_builder

import tensorflow as tf

from tensorflow.python.keras.callbacks import TensorBoard


def train(dataset, model, optimizer):
    train_set, test_set, steps_per_epoch = data_input.get_input(dataset)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    tbCallBack = TensorBoard(log_dir='./logs',
                             histogram_freq=1,
                             write_graph=True,
                             write_grads=True,
                             write_images=True,
                             embeddings_freq=0,
                             embeddings_layer_names=None,
                             embeddings_metadata=None)

    model.fit(train_set,
              epochs=cfg.epochs,
              validation_data=test_set,
              steps_per_epoch=steps_per_epoch,
              verbose=1,
              callbacks=[tbCallBack])


def get_model(model_type, method='dbn', layers=[100], out_num=10, weight_decay=0, input_height=28, input_width=28, input_depth=1):
    if model_type == 'mlp':
        model = model_builder.build_mlp(method, layers, out_num, weight_decay, height=input_height, width=input_width, depth=input_depth)
    elif model_type == 'cnn':
        model = model_builder.build_cnn()
    else:
        model = model_builder.build_cnn()
    return model


def get_optimizer(method, lr, momentum=0.9):
    if method == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr, momentum=momentum)
    elif method == 'adam':
        optimizer = tf.train.AdamOptimizer(lr)
    else:
        optimizer = tf.train.AdamOptimizer(lr)
    return optimizer


def main(_):
    if cfg.strategy == 'debug':
        methods = ['plain', 'dbn', 'dbn_iter']
        lrs = [0.1, 0.5, 1, 5]
        layer_models = [[100], [100,100,100]]
        cfg.epochs = 1000
        cfg.batch_size = 50000
        for method in methods:
            for lr in lrs:
                for layer_model in layer_models:
                    print('method:{}'.format(method))
                    print('lr:{}'.format(lr))
                    print('layer:{}'.format(layer_model))
                    optimizer = get_optimizer('sgd', lr, momentum=0)
                    model = get_model('mlp', method=method, layers=layer_model, weight_decay=0, input_height=28, input_width=28, input_depth=1)
                    train('mnist', model, optimizer)


if __name__ == "__main__":
    tf.app.run()

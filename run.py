import data_input
import model_builder
import os

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from config import cfg
from tensorflow.python.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.backend import clear_session


def train(dataset, model, optimizer, tensorboard=False, learning_rate_scheduler=None):
    train_set, test_set, steps_per_epoch, validation_steps = data_input.get_input(dataset)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    callbacks = []
    if tensorboard:
        callbacks.append(TensorBoard(log_dir='./logs',
                                     histogram_freq=1,
                                     write_graph=True,
                                     write_grads=True,
                                     write_images=True,
                                     embeddings_freq=0,
                                     embeddings_layer_names=None,
                                     embeddings_metadata=None))
    if learning_rate_scheduler:
        callbacks.append(learning_rate_scheduler)

    history = model.fit(train_set,
                        epochs=cfg.epochs,
                        validation_data=test_set,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=callbacks,
                        verbose=1)

    return history


def get_model(model_type, method='dbn', layers=[100], filters=64, out_num=10, weight_decay=0., input_height=28, input_width=28, input_depth=1, m_per_group=0, affine=True):
    if model_type == 'mlp':
        model = model_builder.build_mlp(method, layers, out_num, weight_decay, height=input_height, width=input_width, depth=input_depth, m_per_group=m_per_group, dbn_affine=affine)
    elif model_type == 'vggA':
        model = model_builder.build_vgg(method, filters, [1,1,2,2,2], out_num, weight_decay, height=input_height, width=input_width, depth=input_depth, m_per_group=m_per_group, dbn_affine=affine)

    return model


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
        methods = ['dbn', 'iter_norm', 'bn']
        lrs = [0.1, 0.5, 1, 5]
        layer_models = [[100, 100, 100], [100]]
        cfg.epochs = 3
        cfg.batch_size = 1024
        if os.path.exists(cfg.result + '/debug.csv'):
            os.remove(cfg.result + '/debug.csv')
        df = pd.DataFrame()
        for lr in lrs:
            for layer_model in layer_models:
                fig_acc = plt.figure(num='fig_acc')
                fig_loss = plt.figure(num='fig_loss')
                legends = []
                for method in methods:
                    clear_session()
                    print('method:{}'.format(method))
                    print('lr:{}'.format(lr))
                    print('layer:{}'.format(layer_model))
                    optimizer = get_optimizer('sgd', lr, momentum=0)
                    model = get_model('mlp', method=method, layers=layer_model, weight_decay=0, input_height=28, input_width=28, input_depth=1)
                    plot_name = '-'.join([str(i) for i in layer_model]) + '_' + '_'.join([method, str(lr)])
                    history = train('mnist', model, optimizer)
                    df[plot_name + '_acc'] = history.history['acc']
                    df[plot_name + '_val_acc'] = history.history['val_acc']
                    plt.figure(num='fig_acc')
                    plt.plot(history.history['acc'])
                    plt.plot(history.history['val_acc'])
                    plt.title('model accuracy')
                    plt.ylabel('accuracy')
                    plt.xlabel('epoch')

                    df[plot_name + '_loss'] = history.history['loss']
                    df[plot_name + '_val_loss'] = history.history['val_loss']
                    plt.figure(num='fig_loss')
                    plt.plot(history.history['loss'])
                    plt.plot(history.history['val_loss'])
                    plt.title('model loss')
                    plt.ylabel('loss')
                    plt.xlabel('epoch')
                    plt.legend([method + '_train', method + '_test'], loc='upper left')

                    legends += [method + '_train', method + '_test']

                plt.figure(num='fig_acc')
                plt.legend(legends, loc = 'upper left')
                plt.savefig(cfg.result + '/acc_' + '-'.join([str(i) for i in layer_model]) + '_' + str(lr) + '.jpg')
                plt.close(fig=fig_acc)
                plt.figure(num='fig_loss')
                plt.legend(legends, loc = 'upper left')
                plt.savefig(cfg.result + '/loss_' + '-'.join([str(i) for i in layer_model]) + '_' + str(lr) + '.jpg')
                plt.close(fig=fig_loss)
        df.to_csv(cfg.result + '/debug.csv')

    elif cfg.strategy == 'vggA_base':
        methods = ['dbn', 'iter_norm', 'bn']
        lr = 4
        cfg.epochs = 3
        cfg.batch_size = 256
        cfg.augment = True
        if os.path.exists(cfg.result + '/vggA_base.csv'):
            os.remove(cfg.result + '/vggA_base.csv')
        df = pd.DataFrame()
        fig_acc = plt.figure(num='fig_acc')
        fig_loss = plt.figure(num='fig_loss')
        legends = []
        for method in methods:
            clear_session()
            print('method:{}'.format(method))
            print('lr:{}'.format(lr))
            optimizer = get_optimizer('sgd', lr, momentum=0.9)

            def lr_scheduler(epoch, lr):
                decay_rate = 0.5
                decay_step = 20
                if epoch % decay_step == 0 and epoch:
                    return lr * decay_rate
                return lr

            model = get_model('vggA', method=method, filters=4, weight_decay=0.0005, input_height=32,
                              input_width=32, input_depth=3, m_per_group=16, affine=True)
            plot_name = '_'.join([method, str(lr)])
            history = train('cifar10', model, optimizer, learning_rate_scheduler=LearningRateScheduler(lr_scheduler))
            df[plot_name + '_acc'] = history.history['acc']
            df[plot_name + '_val_acc'] = history.history['val_acc']
            plt.figure(num='fig_acc')
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            df[plot_name + '_loss'] = history.history['loss']
            df[plot_name + '_val_loss'] = history.history['val_loss']
            plt.figure(num='fig_loss')
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend([method + '_train', method + '_test'], loc='upper left')
            legends += [method + '_train', method + '_test']

        plt.figure(num='fig_acc')
        plt.legend(legends, loc='upper left')
        plt.savefig(cfg.result + '/vggA_base_acc_' + str(lr) + '.jpg')
        plt.close(fig=fig_acc)
        plt.figure(num='fig_loss')
        plt.legend(legends, loc='upper left')
        plt.savefig(cfg.result + '/vggA_base_loss_' + str(lr) + '.jpg')
        plt.close(fig=fig_loss)
        df.to_csv(cfg.result + '/vggA_base.csv')


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    tf.app.run()

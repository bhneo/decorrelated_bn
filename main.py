import argparse
import os
from importlib import import_module
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.keras import backend as K
import tensorflow as tf
import data_input

from config import config, build_config


is_tracing = False


def get_extra_losses(model):
    loss = 0
    if len(model.losses) > 0:
        loss = tf.math.add_n(model.losses)
    return loss


def get_train_step(model, loss, accuracy):
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            extra_loss = get_extra_losses(model)
            pred_loss = model.loss(labels, predictions)
            total_loss = pred_loss + extra_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # Update the metrics
        loss.update_state(total_loss)
        accuracy.update_state(labels, predictions)
    return train_step


def get_test_step(model, loss, accuracy):
    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        extra_loss = get_extra_losses(model)
        pred_loss = model.loss(labels, predictions)
        total_loss = pred_loss + extra_loss
        # Update the metrics
        loss.update_state(total_loss)
        accuracy.update_state(labels, predictions)
    return test_step


def get_log_step(tensor_log, writer):

    @tf.function
    def log_step(images, labels, epoch):
        if tensor_log:
            logs = tensor_log.model(images)
            with writer.as_default():
                tensor_log.summary(logs, epoch)
    return log_step


def do_callbacks(state, callbacks, epoch=0, batch=0):
    if state == 'on_train_begin':
        for callback in callbacks:
            callback.on_train_begin()
    if state == 'on_epoch_begin':
        for callback in callbacks:
            callback.on_epoch_begin(epoch)
    if state == 'on_epoch_end':
        for callback in callbacks:
            callback.on_epoch_end(epoch)
    if state == 'on_batch_begin':
        for callback in callbacks:
            callback.on_batch_begin(batch)
    if state == 'on_batch_end':
        for callback in callbacks:
            callback.on_batch_end(batch)


def log_trace(step, writer, logdir):
    global is_tracing
    if step == 1 and not is_tracing:
        summary_ops_v2.trace_on(graph=True, profiler=True)
        is_tracing = True
        print('start tracing...')
    elif is_tracing:
        with writer.as_default():
            summary_ops_v2.trace_export(
                name='Default',
                step=step,
                profiler_outdir=os.path.join(logdir, 'train'))
        is_tracing = False
        print('export trace!')


def train(model, tensor_log, manager, init_epoch, train_set, test_set):
    logdir = os.path.join(config.logdir, model.name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    train_writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
    test_writer = tf.summary.create_file_writer(os.path.join(logdir, 'test'))

    with train_writer.as_default():
        summary_ops_v2.graph(K.get_graph(), step=0)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    train_step = get_train_step(model, train_loss, train_accuracy)
    test_step = get_test_step(model, test_loss, test_accuracy)
    log_step = get_log_step(tensor_log, train_writer)

    do_callbacks('on_train_begin', model.callbacks)
    for epoch in range(init_epoch, config.training.epochs):
        do_callbacks('on_epoch_begin', model.callbacks, epoch=epoch)
        # Reset the metrics
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        tf.keras.backend.set_learning_phase(1)
        for batch, (images, labels) in enumerate(train_set):
            do_callbacks('on_batch_begin', model.callbacks, batch=batch)
            train_step(images, labels)
            if batch == 0 and config.training.log:
                log_step(images, labels, epoch)
            do_callbacks('on_batch_end', model.callbacks, batch=batch)
        do_callbacks('on_epoch_end', model.callbacks, epoch=epoch)
        # Get the metric results
        train_loss_result = float(train_loss.result())
        train_accuracy_result = float(train_accuracy.result())
        with train_writer.as_default():
            tf.summary.scalar('loss', train_loss_result, step=epoch+1)
            tf.summary.scalar('accuracy', train_accuracy_result, step=epoch+1)

        # Run a test loop at the end of each epoch.
        tf.keras.backend.set_learning_phase(0)
        for images, labels in test_set:
            test_step(images, labels)
        # Get the metric results
        test_loss_result = float(test_loss.result())
        test_accuracy_result = float(test_accuracy.result())
        with test_writer.as_default():
            tf.summary.scalar('loss', test_loss_result, step=epoch+1)
            tf.summary.scalar('accuracy', test_accuracy_result, step=epoch+1)

        print('Epoch:{}, train acc:{:f}, test acc:{:f}'.format(epoch+1, train_accuracy_result, test_accuracy_result))
        if (config.training.save_frequency != 0 and epoch % config.training.save_frequency == 0) or epoch == config.training.epochs-1:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(model.optimizer.iterations.numpy(), save_path))


def evaluate(model, test_set):
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    test_step = get_test_step(model, test_loss, test_accuracy)
    # Run a test loop at the end of each epoch.
    print('learning phase:', tf.keras.backend.learning_phase())
    for images, labels in test_set:
        test_step(images, labels)
    # Get the metric results
    test_loss_result = test_loss.result()
    test_accuracy_result = test_accuracy.result()

    print('test loss:{:f}, test acc:{:f}'.format(test_loss_result, test_accuracy_result))


def main(arguments):
    print(os.getcwd())
    train_set, test_set, info = data_input.build_dataset(config.dataset.name, batch_size=config.training.batch_size, flip=arguments.flip, crop=arguments.crop)
    model, tensor_log = import_module('models.' + config.model.name).build_model(shape=info.features['image'].shape,
                                                                                 num_out=info.features['label'].num_classes)
    progress = tf.keras.callbacks.ProgbarLogger('steps')
    progress.set_params({'verbose': True,
                         'epochs': int(arguments.epochs),
                         'metrics': '',
                         'steps': 1 + info.splits['train'].num_examples // config.training.batch_size})
    model.callbacks.append(progress)

    config.logdir = os.path.join(config.logdir, config.dataset.name)
    print('config:', config)
    model_dir = os.path.join(config.logdir, model.name)

    ckpt = tf.train.Checkpoint(optimizer=model.optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        init_epoch = config.training.batch_size * model.optimizer.iterations.numpy() // info.splits['train'].num_examples
    else:
        print("Initializing from scratch.")
        init_epoch = 0

    if arguments.train:
        train(model, tensor_log, manager, init_epoch, train_set, test_set)
    else:
        evaluate(model, test_set)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    parser.add_argument('--train', default=True, help='train of evaluate')
    parser.add_argument('--t_log', default=True, help='tensorboard log')
    parser.add_argument('--dataset', default=config.dataset.name, help='dataset config')
    parser.add_argument('--flip', default=config.dataset.flip, help='dataset config')
    parser.add_argument('--crop', default=config.dataset.crop, help='dataset config')
    parser.add_argument('--model', default=config.model.name, help='network config')
    parser.add_argument('--idx', default=1, help='the index of trial')
    parser.add_argument('--epochs', default=config.training.epochs, help='the total training epochs')
    parser.add_argument('--batch', default=config.training.batch_size, help='the training batch_size')
    parser.add_argument('--lr', default=config.training.lr, help='learning rate')
    parser.add_argument('--steps', default=config.training.steps, help='the total training steps')
    parser.add_argument('--log', default=config.logdir, help='directory to save log')
    parser.add_argument('--log_steps', default=config.training.log_steps, help='frequency to log by steps')
    parser.add_argument('--layer_num', default=config.model.layer_num, help='the number of layers')
    parser.add_argument('--normalize', default=config.normalize.type, help='the type of bn')
    parser.add_argument('--dbn_m', default=config.normalize.m, help='m per group in dbn')
    parser.add_argument('--iter', default=config.normalize.iter, help='iter number')
    parser.add_argument('--dbn_affine', default=config.normalize.affine, help='affine or not')
    arguments = parser.parse_args()
    build_config(arguments)
    return arguments


if __name__ == "__main__":
    args = parse_args()
    main(args)

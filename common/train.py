import os

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import summary_ops_v2


class Trainer(object):
    def __init__(self, model, params, data_info, monitor=None, bar=True, inference_label=False, max_save=1):
        self.model = model
        self.params = params
        self.data_info = data_info
        self.monitor = monitor
        self.bar = bar
        self.max_save = max_save
        self.inference_label = inference_label
        self.metrics = {}
        self.add_metrics()
        self.manager, self.init_epoch = self.init()

    def init(self):
        if self.bar:
            progress = tf.keras.callbacks.ProgbarLogger('steps')
            progress.set_params({'verbose': True,
                                 'epochs': int(self.params.training.epochs),
                                 'metrics': '',
                                 'steps': 1 + self.data_info.splits['train_examples'] // self.params.training.batch_size})
            self.model.callbacks.append(progress)

        self.params.logdir = os.path.join(self.params.logdir, self.params.dataset.name)
        model_dir = os.path.join(self.params.logdir, self.model.name)

        ckpt = tf.train.Checkpoint(optimizer=self.model.optimizer, net=self.model)
        manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=self.max_save)
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        init_epoch = self.params.training.batch_size * self.model.optimizer.iterations.numpy() // self.data_info.splits['train_examples']
        return manager, init_epoch

    def extra_loss(self):
        loss = 0
        if len(self.model.losses) > 0:
            loss = tf.math.add_n(self.model.losses)
        return loss

    def add_metrics(self):
        self.metrics['loss'] = tf.keras.metrics.Mean(name='loss')
        self.metrics['accuracy'] = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    def reset_metrics(self):
        for metric in self.metrics:
            self.metrics[metric].reset_states()

    def update_metrics(self, loss, labels, predictions):
        for name in self.metrics:
            if name == 'loss':
                self.metrics[name].update_state(loss)
            else:
                self.metrics[name].update_state(labels, predictions)

    def get_loss(self, labels, predictions):
        extra_loss = self.extra_loss()
        pred_loss = self.model.loss(labels, predictions)
        total_loss = pred_loss + extra_loss
        return total_loss

    def do_inference(self, inputs, labels):
        if self.inference_label:
            prediction, recons = self.model((inputs, labels))
            return prediction
        else:
            return self.model(inputs)

    def get_train_step(self):
        @tf.function
        def train_step(inputs, labels):
            with tf.GradientTape() as tape:
                outputs = self.do_inference(inputs, labels)
                loss = self.get_loss(labels, outputs)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss, outputs
        return train_step

    def get_test_step(self):
        @tf.function
        def test_step(inputs, labels):
            outputs = self.do_inference(inputs, labels)
            loss = self.get_loss(labels, outputs)
            return loss, outputs
        return test_step

    def log_tensors(self, batch, inputs, labels, writer, epoch):
        if batch == 0 and self.monitor is not None:
            logs = self.monitor.model((inputs, labels))
            with writer.as_default():
                self.monitor.summary(logs, epoch)

    def metrics_results(self, prefix, writer=None, epoch=0):
        print_str = prefix
        for name in self.metrics:
            result = float(self.metrics[name].result())
            print_str += '  {}:{:f}'.format(name, result)
            if writer is not None:
                with writer.as_default():
                    tf.summary.scalar(name, result, step=epoch + 1)
        print(print_str)

    def checkpoint(self, manager, step, frequency):
        if (frequency != 0 and step % frequency == 0) or frequency == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(self.model.optimizer.iterations.numpy(), save_path))

    def evaluate(self, test_set, test_writer=None, epoch=0):
        tf.keras.backend.set_learning_phase(0)
        test_step = self.get_test_step()
        self.reset_metrics()
        for inputs, labels in test_set:
            loss, outputs = test_step(inputs, labels)
            self.update_metrics(loss, labels, outputs)
        self.metrics_results('Test: ', test_writer, epoch)

    def train_one_epoch(self, train_set, train_writer, epoch):
        do_callbacks('on_epoch_begin', self.model.callbacks, epoch=epoch)
        tf.keras.backend.set_learning_phase(1)
        train_step = self.get_train_step()
        self.reset_metrics()
        for batch, (inputs, labels) in enumerate(train_set):
            do_callbacks('on_batch_begin', self.model.callbacks, batch=batch)
            loss, outputs = train_step(inputs, labels)
            self.update_metrics(loss, labels, outputs)
            self.log_tensors(batch, inputs, labels, train_writer, epoch)
            do_callbacks('on_batch_end', self.model.callbacks, batch=batch)
        do_callbacks('on_epoch_end', self.model.callbacks, epoch=epoch)
        self.metrics_results('Train: ', train_writer, epoch)

    def fit(self, train_set, test_set, by_epoch=True):
        logdir = os.path.join(self.params.logdir, self.model.name)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        train_writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
        test_writer = tf.summary.create_file_writer(os.path.join(logdir, 'test'))

        with train_writer.as_default():
            summary_ops_v2.graph(K.get_graph(), step=0)

        do_callbacks('on_train_begin', self.model.callbacks)
        for epoch in range(self.init_epoch, self.params.training.epochs):
            self.train_one_epoch(train_set, train_writer, epoch)
            self.evaluate(test_set, test_writer, epoch)
            self.checkpoint(self.manager, epoch+1, self.params.training.save_frequency)
        self.checkpoint(self.manager, self.params.training.epochs, 0)


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

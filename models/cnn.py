import tensorflow as tf
from tqdm import tqdm
import module.DecorelateBN_PowerIter as dbn
from models import data_input


flags = tf.app.flags


############################
#    hyper parameters      #
############################

flags.DEFINE_integer('batch_size', 1024, 'batch size')
flags.DEFINE_integer('steps', 10000, 'steps')
flags.DEFINE_integer('save_summaries_steps', 10, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('save_checkpoint_steps', 200, 'the frequency of saving model')
flags.DEFINE_list('layers', [100, 100, 100, 100], 'mlp layers')
flags.DEFINE_float('lr', 0.1, 'learning rate')

############################
#   environment setting    #
############################
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_string('logdir', 'cnn_logdir_1_1024_fashion', 'logs directory')
flags.DEFINE_string('mode', 'dbn', 'plain:nothing inserted, bn: batch normalization in tf, dbn: decorrelated batch normalization')
flags.DEFINE_string('data', 'fashion-mnist', 'data set...')

cfg = tf.app.flags.FLAGS


def train():
    handle = tf.placeholder(tf.string, [])
    X, y, train_iterator, val_iterator, num_label, num_batch = data_input.create_train_set(cfg.data, handle, cfg.batch_size)
    y = tf.one_hot(y, depth=num_label, axis=-1, dtype=tf.float32)
    is_training = tf.placeholder(tf.bool, shape=[])
    summary = []

    layer = X
    for i in range(len(cfg.layers)):
        layer = tf.layers.dense(layer, cfg.layers[i], activation=None, name='layer{}'.format(i))
        if cfg.mode == 'plain':
            pass
        elif cfg.mode == 'bn':
            layer = tf.layers.batch_normalization(layer, training=is_training)
        elif cfg.mode == 'dbn':
            layer = dbn.buildDBN(layer, is_training, summary)
        layer = tf.nn.relu(layer)
        summary.append(tf.summary.histogram('layer{}'.format(i), layer))

    logits = tf.layers.dense(layer, num_label, activation=None)
    outputs = tf.nn.softmax(logits)
    summary.append(tf.summary.histogram('outputs', outputs))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    summary.append(tf.summary.scalar('loss', loss))
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = tf.train.GradientDescentOptimizer(cfg.lr).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    summary.append(tf.summary.scalar('accuracy', accuracy))
    merged_summary = tf.summary.merge(summary)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(cfg.logdir + '/train_' + cfg.mode, sess.graph)
        valid_writer = tf.summary.FileWriter(cfg.logdir + '/valid_' + cfg.mode)

        train_handle = sess.run(train_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())

        # train
        bar = tqdm(range(cfg.steps), ncols=70, leave=False, unit='b')
        for i in bar:
            if i % cfg.save_summaries_steps == 0 and i != 0:
                train_loss, train_acc, train_summary = sess.run([loss, accuracy, merged_summary],
                                                                feed_dict={handle: train_handle,
                                                                           is_training: False})
                train_writer.add_summary(train_summary, i)

                valid_loss, valid_acc, valid_summary = sess.run([loss, accuracy, merged_summary],
                                                                feed_dict={handle: val_handle,
                                                                           is_training: False})
                valid_writer.add_summary(valid_summary, i)
            else:
                sess.run(train_op, feed_dict={handle: train_handle, is_training: True})
        bar.close()

    train_writer.close()
    valid_writer.close()


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
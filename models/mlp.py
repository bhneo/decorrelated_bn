import tensorflow as tf
from tqdm import tqdm
import module.DecorelateBN_PowerIter as dbn
from models import mnist_input


flags = tf.app.flags


############################
#    hyper parameters      #
############################

flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('steps', 10000, 'steps')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_integer('save_summaries_steps', 10, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('save_checkpoint_steps', 200, 'the frequency of saving model')
flags.DEFINE_list('layers', [100, 100, 100, 100], 'mlp layers')
flags.DEFINE_float('lr', 0.01, 'learning rate')

############################
#   environment setting    #
############################
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_string('mode', 'plain', 'plain:nothing inserted, bn: batch normalization in tf, dbn: decorrelated batch normalization')

cfg = tf.app.flags.FLAGS


def get_batch(X, y, idx, batch_size, batch_num):
    if idx


def train():
    trX, trY, num_tr_batch, valX, valY, num_val_batch = mnist_input.load_mnist('mnist')

    # inputs
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    is_training = tf.placeholder(tf.bool, shape=[])
    summary = []

    for i in cfg.layers:
        layer = tf.layers.dense(x, 100, activation=None)
        if cfg.mode == 'plain':
            pass
        elif cfg.mode == 'bn':
            layer = tf.layers.batch_normalization(layer, training=is_training)
        elif cfg.mode == 'dbn':
            layer = dbn.buildDBN(layer, is_training)
        layer = tf.nn.sigmoid(layer)
        summary.append(tf.summary.histogram('layer{}'.format(i), layer))

    logits = layer
    outputs = tf.nn.softmax(logits)
    summary.append(tf.summary.histogram('outputs', outputs))
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    summary.append(tf.summary.scalar('loss', loss))
    train_op = tf.train.GradientDescentOptimizer(cfg.lr).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    summary.append(tf.summary.scalar('accuracy', accuracy))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(cfg.logdir + '/train_' + cfg.mode, sess.graph)
        valid_writer = tf.summary.FileWriter(cfg.logdir + '/valid_' + cfg.mode)

        # train
        bar = tqdm(range(cfg.steps), ncols=70, leave=False, unit='b')
        for i in bar:
            if i % cfg.save_summaries_steps == 0:
                train_batch = mnist.validation.next_batch(cfg.batch_size)
                train_loss, train_acc, train_summary = sess.run([loss, accuracy, summary], feed_dict={x: train_batch[0], y: train_batch[1]})
                train_writer.add_summary(train_summary, i)

                valid_batch = mnist.validation.next_batch(cfg.batch_size)
                valid_loss, valid_acc, valid_summary = sess.run([loss, accuracy, summary],
                                                                feed_dict={x: valX, y: valY})
                valid_writer.add_summary(valid_summary, i)
            else:
                batch = mnist.train.next_batch(cfg.batch_size)
                sess.run(train_op, feed_dict={x: batch[0], y: batch[1]})
        bar.close()

    train_writer.close()
    valid_writer.close()


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
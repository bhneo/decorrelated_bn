from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


flags = tf.app.flags


############################
#    hyper parameters      #
############################

# for training
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('epoch', 1, 'epoch')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_integer('save_summaries_steps', 10, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('save_checkpoint_steps', 200, 'the frequency of saving model')

############################
#   environment setting    #
############################
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')

cfg = tf.app.flags.FLAGS


def train():
    # MNIST数据存放的路径
    file = "./MNIST"

    # 导入数据
    mnist = input_data.read_data_sets(file, one_hot=True)

    # 模型的输入和输出
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # 模型的权重和偏移量
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # 创建Session
    sess = tf.InteractiveSession()
    # 初始化权重变量
    sess.run(tf.global_variables_initializer())

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # 交叉熵
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    # 训练
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    # 测试
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def evaluate():
    pass



def main(_):
    if cfg.model == 'vector':
        from models.vector_caps_model import CapsNet as Model
    elif cfg.model == 'matrix':
        from models.matrix_caps_model import CapsNet as Model
    else:
        from models.baseline import Model

    model = Model()

    if cfg.is_training:
        tf.logging.info(' Start training...')
        train(model)
        tf.logging.info('Training done')
    else:
        evaluation(model)

if __name__ == "__main__":
    tf.app.run()
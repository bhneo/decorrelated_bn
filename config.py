import tensorflow as tf


flags = tf.flags
############################
#    hyper parameters      #
############################
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_integer('epochs', 100, 'steps')
flags.DEFINE_integer('save_summaries_steps', 10, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('save_checkpoint_steps', 200, 'the frequency of saving model')
flags.DEFINE_list('layers', [100, 100, 100, 100], 'mlp layers')
flags.DEFINE_float('lr', 1, 'learning rate')
flags.DEFINE_string('activation', 'relu', 'activation used in hidden layers')

############################
#   environment setting    #
############################
flags.DEFINE_string('result', 'result', 'logs directory')
flags.DEFINE_string('mode', 'plain', 'plain:nothing inserted, bn: batch normalization in tf, dbn: decorrelated batch normalization')
flags.DEFINE_string('dataset', 'fashion-mnist', 'data set...')
flags.DEFINE_string('strategy', 'vgg16', '')
flags.DEFINE_boolean('augment', False, 'do data augment or not')
flags.DEFINE_string('data_set_path', 'data/imagenet/', 'dataset path in which dataset saved')

cfg = tf.flags.FLAGS
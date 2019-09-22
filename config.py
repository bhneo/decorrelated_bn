from easydict import EasyDict
from common import utils


config = EasyDict()

config.logdir = 'log'  # dir of logs

config.dataset = EasyDict()
config.dataset.name = 'multi_mnist'
config.dataset.flip = True
config.dataset.crop = True

config.training = EasyDict()
config.training.batch_size = 128
config.training.epochs = 163
config.training.steps = 9999999  # The number of training steps
config.training.lr = 0.1
config.training.lr_steps = [30000, 40000]
config.training.verbose = True
config.training.log_steps = 1000
config.training.idx = 1
config.training.momentum = 0.9
config.training.save_frequency = 0
config.training.log = True

config.model = EasyDict()
config.model.name = 'vgg'  # which model to use
config.model.layer_num = 11

config.normalize = EasyDict()
config.normalize.type = 'dbn'
config.normalize.m = 16
config.normalize.iter = 3
config.normalize.affine = True


def build_config(args):
    config.dataset.name = args.dataset
    config.dataset.flip = utils.str2bool(args.flip)
    config.dataset.crop = utils.str2bool(args.crop)
    config.logdir = args.log
    config.training.log_steps = int(args.log_steps)
    config.training.idx = args.idx
    config.training.epochs = int(args.epochs)
    config.training.lr = float(args.lr)
    config.training.batch_size = int(args.batch)
    config.training.steps = int(args.steps)
    config.training.log = utils.str2bool(args.t_log)
    config.model.name = args.model
    config.model.layer_num = int(args.layer_num)
    config.normalize.type = args.normalize
    config.normalize.m = int(args.dbn_m)
    config.normalize.iter = int(args.iter)
    config.normalize.affine = utils.str2bool(args.dbn_affine)







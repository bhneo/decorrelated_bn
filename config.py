from easydict import EasyDict
from common import utils

import argparse

params = EasyDict()

params.logdir = 'log'  # dir of logs

params.dataset = EasyDict()
params.dataset.name = 'cifar10'
params.dataset.flip = True
params.dataset.crop = True

params.training = EasyDict()
params.training.batch_size = 128
params.training.epoch = 160
params.training.lr = 0.1
params.training.idx = 1
params.training.save_frequency = 10

params.model = EasyDict()
params.model.arch = 'vgg'  # which model to use

params.normalize = EasyDict()
params.normalize.method = 'dbn'
params.normalize.m = 16
params.normalize.iter = 5
params.normalize.affine = True


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--train', default=True, help='train of evaluate')
    parser.add_argument('--log', default=params.logdir, help='path to save logs')
    parser.add_argument('--dataset', default=params.dataset.name, help='dataset config')
    parser.add_argument('--flip', default=params.dataset.flip, help='dataset config')
    parser.add_argument('--crop', default=params.dataset.crop, help='dataset config')
    parser.add_argument('--batch', default=params.training.batch_size, help='the training batch_size')
    parser.add_argument('--epoch', default=params.training.epoch, help='the total training epochs')
    parser.add_argument('--lr', default=params.training.lr, help='initial learning rate')
    parser.add_argument('--idx', default=params.training.idx, help='the index of trial')
    parser.add_argument('--save', default=params.training.save_frequency, help='save frequency')
    parser.add_argument('--arch', default=params.model.arch, help='model architecture')
    parser.add_argument('--method', default=params.normalize.method, help='whiten method')
    parser.add_argument('--m', default=params.normalize.m, help='size of per group')
    parser.add_argument('--iter', default=params.normalize.iter, help='iterations for iter-norm')
    parser.add_argument('--affine', default=params.normalize.affine, help='whether to do affine')
    arguments = parser.parse_args()
    build_params = build_config(arguments, params)
    return arguments, build_params


def build_config(args, build_params):
    build_params.logdir = args.log
    build_params.dataset.name = args.dataset
    build_params.dataset.flip = utils.str2bool(args.flip)
    build_params.dataset.crop = utils.str2bool(args.crop)
    build_params.training.batch_size = int(args.batch)
    build_params.training.epoch = int(args.epoch)
    build_params.training.lr = float(args.lr)
    build_params.training.idx = args.idx
    build_params.training.save_frequency = args.save
    build_params.model.arch = args.arch
    build_params.normalize.method = args.method
    build_params.normalize.m = int(args.m)
    build_params.normalize.iter = int(args.iter)
    build_params.normalize.affine = utils.str2bool(args.affine)
    return build_params







import os
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from common import utils


class DataInfo(object):
    def __init__(self,
                 features=None,
                 splits=None):
        if features and not isinstance(features, tfds.features.FeaturesDict):
            raise TypeError('not the instance of tensorflow_dataset.features.FeaturesDict')
        self.features = features
        self.splits = splits

    def from_tfds_info(self, dataset_info):
        if not isinstance(dataset_info, tfds.core.DatasetInfo):
            raise TypeError('not the instance of tensorflow_dataset.core.DatasetInfo')
        self.features = dataset_info.features
        self.splits = {'train_examples': dataset_info.splits['train'].num_examples,
                       'test_examples': dataset_info.splits['test'].num_examples}
        return self


def download_and_check(name):
    dataset, info = tfds.load(name, data_dir=name, with_info=True)
    print(info.features['image'].shape)
    print(info.features['label'].num_classes)
    print(info.splits['train'].num_examples)
    print(info.splits['test'].num_examples)
    print(info)


def build_parse(height, width, channel, image_standardization=True, flip=True, crop=True, brightness=False, contrast=False):
    image_standardization = utils.str2bool(image_standardization)
    flip = utils.str2bool(flip)
    crop = utils.str2bool(crop)
    brightness = utils.str2bool(brightness)
    contrast = utils.str2bool(contrast)

    def parse(image, label):
        image = tf.cast(image, tf.float32)
        if image_standardization:
            image = tf.image.per_image_standardization(image)
        else:
            image = tf.divide(image, 255.)
        if flip:
            image = tf.image.random_flip_left_right(image)
        if crop:
            image = tf.image.resize_with_crop_or_pad(image, height+8, width+8)
            image = tf.image.random_crop(image, [height, width, channel])
        if brightness:
            image = tf.image.random_brightness(image, max_delta=63)
        if contrast:
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        return image, label
    return parse


def load_data(name):
    if name in tfds.list_builders():
        return tfds.load(name, data_dir=name, with_info=True, as_supervised=True)
    else:
        pass


def build_dataset(name, path='data', parser_train=None, parser_test=None,
                  batch_size=128, buffer_size=50000,
                  image_standardization=True, flip=True, crop=True,
                  brightness=False, contrast=False):
    dataset, info = tfds.load(name, data_dir=os.path.join(path, name), with_info=True, as_supervised=True)
    info = DataInfo().from_tfds_info(info)
    train = dataset['train']
    test = dataset['test']
    image_height, image_width, image_channel = info.features['image'].shape[0], info.features['image'].shape[1], info.features['image'].shape[2]

    if buffer_size > 0:
        train = train.shuffle(buffer_size=buffer_size)

    if parser_train is None:
        parser_train = build_parse(image_height,
                                   image_width,
                                   image_channel,
                                   image_standardization=image_standardization,
                                   flip=flip,
                                   crop=crop,
                                   brightness=brightness,
                                   contrast=contrast)
    train = train.map(parser_train,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    if parser_test is None:
        parser_test = build_parse(image_height,
                                  image_width,
                                  image_channel,
                                  image_standardization=image_standardization,
                                  flip=False,
                                  crop=False,
                                  brightness=False,
                                  contrast=False)
    test = test.map(parser_test,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return train, test, info


def count_data(name, path='data'):
    train, test, info = build_dataset(name, path)
    train_num = 0
    for image, label in train:
        train_num += image.shape[0]

    test_num = 0
    for image, label in test:
        test_num += image.shape[0]

    print('train num:', train_num)
    print('test num:', test_num)


def view_data(name, path='data', img_stand=False):
    train, test, info = build_dataset(name, path, image_standardization=img_stand)
    for image, label in train:
        if not img_stand:
            image /= 255.
        out_image(image, label)
        break

    for image, label in test:
        if not img_stand:
            image /= 255.
        out_image(image, label)
        break


def out_image(images, labels):
    plt.figure()
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.title(labels[i].numpy())
        image = images[i, :, :, :]
        if image.shape[-1] == 1:
            image = np.squeeze(image, -1)
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
    plt.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == "__main__":
    # download_and_check('cifar10')
    # count_data('cifar10')
    view_data('fashion_mnist', '', False)


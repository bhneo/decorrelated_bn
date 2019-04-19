import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import cfg


def load_data_on_memory(dataset):
    if dataset == 'mnist':
        data = keras.datasets.mnist
    elif dataset == 'fashion_mnist':
        data = keras.datasets.fashion_mnist
    elif dataset == 'cifar10':
        data = keras.datasets.cifar10
    elif dataset == 'cifar100':
        data = keras.datasets.cifar100
    else:
        data = keras.datasets.cifar10

    (train_images, train_labels), (test_images, test_labels) = data.load_data()

    if len(train_images.shape) == 3:
        train_images = np.expand_dims(train_images, -1)

    if len(test_images.shape) == 3:
        test_images = np.expand_dims(test_images, -1)

    return train_images, train_labels, test_images, test_labels


def get_input(dataset):
    if dataset in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
        train_images, train_labels, test_images, test_labels = load_data_on_memory(dataset)
        image_height, image_width, image_depth = train_images.shape[1], train_images.shape[2], train_images.shape[3]

        def parse_train(image, label):
            image = tf.image.per_image_standardization(image)
            if cfg.augment:
                image = tf.image.random_flip_left_right(image)
                image = tf.image.resize_image_with_crop_or_pad(image, image_height + 8, image_width + 8)
                image = tf.random_crop(image, [image_height, image_width, image_depth])
            return image, label

        def parse_test(image, label):
            return tf.image.per_image_standardization(image), label

        train_set = tf.data.Dataset.from_tensor_slices((train_images, train_labels))\
            .map(parse_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat().batch(cfg.batch_size)
        test_set = tf.data.Dataset.from_tensor_slices((test_images, test_labels))\
            .map(parse_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat().batch(cfg.batch_size)
        steps_per_epoch = train_images.shape[0]//cfg.batch_size
        validation_steps = test_images.shape[0]//cfg.batch_size
        if validation_steps == 0:
            validation_steps = 1
    elif dataset == 'imagenet':
        def parse_tfrecords(example_proto):
            features = {"image": tf.FixedLenFeature([], tf.string, default_value=""),
                        "height": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                        "width": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                        "label": tf.FixedLenFeature([1], tf.int64, default_value=[0])
                        }
            parsed_features = tf.parse_single_example(example_proto, features)
            image, image_height, image_width, label = parsed_features['image'], parsed_features['height'], \
                                                      parsed_features['width'], parsed_features['label']
            image_decoded = tf.cast(tf.image.decode_jpeg(image, channels=3), tf.float32)
            return image_decoded, image_height, image_width, label

        def parse_train(example_proto):
            image_decoded, image_height, image_width, label = parse_tfrecords(example_proto)
            if cfg.augment:
                random_s = tf.random_uniform([1], minval=256, maxval=481, dtype=tf.int32)[0]
                resized_height, resized_width = tf.cond(image_height < image_width,
                                                        lambda: (random_s, tf.cast(
                                                            tf.multiply(tf.cast(image_width, tf.float64),
                                                                        tf.divide(random_s, image_height)), tf.int32)),
                                                        lambda: (tf.cast(tf.multiply(tf.cast(image_height, tf.float64),
                                                                                     tf.divide(random_s, image_width)),
                                                                         tf.int32), random_s))

                image_resized = tf.image.resize_images(image_decoded, [resized_height, resized_width])
                image_flipped = tf.image.random_flip_left_right(image_resized)
                image_cropped = tf.random_crop(image_flipped, [224, 224, 3])
                image = tf.image.random_brightness(image_cropped, max_delta=63)
                image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
            else:
                image = tf.image.resize_images(image_decoded, [224, 224])

            image = tf.image.per_image_standardization(image)
            return image, label

        def parse_test(example_proto):
            image_decoded, image_height, image_width, label = parse_tfrecords(example_proto)
            image = tf.image.resize_images(image_decoded, [224, 224])
            return tf.image.per_image_standardization(image), label

        train_files_names = os.listdir(cfg.data_set_path + 'train/')
        train_files = [cfg.data_set_path + 'train/' + item for item in train_files_names]
        train_set = tf.data.TFRecordDataset(train_files)\
            .map(parse_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .batch(cfg.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        test_files_names = os.listdir(cfg.data_set_path + 'valid/')
        test_files = [cfg.data_set_path + 'valid/' + item for item in test_files_names]
        test_set = tf.data.TFRecordDataset(test_files) \
            .map(parse_test, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(cfg.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        steps_per_epoch = 1281167 // cfg.batch_size
        validation_steps = 50000 // cfg.batch_size
    else:
        raise ValueError('Dataset not supported yet:', dataset)

    return train_set, test_set, steps_per_epoch, validation_steps


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_data_on_memory('mnist')
    if train_images.shape[-1] == 1:
        train_images = train_images[:, :, :, 0]
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if len(train_images.shape) == 3:
            plt.imshow(train_images[i], cmap='gray')
        else:
            plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.show()

import tensorflow as tf


class TensorLog(object):
    def __init__(self):
        self.hist = {}
        self.scalar = {}
        self.image = {}
        self.model = None

    def add_hist(self, name, tensor):
        self.hist[name] = tensor

    def add_scalar(self, name, tensor):
        self.scalar[name] = tensor

    def add_image(self, name, image):
        self.image[name] = image

    def get_outputs(self):
        outputs = []
        for key in self.hist:
            outputs.append(self.hist[key])
        for key in self.scalar:
            outputs.append(self.scalar[key])
        for key in self.image:
            outputs.append(self.image[key])
        return outputs

    def set_model(self, model):
        self.model = model

    def summary(self, outputs, epoch):
        i = 0
        for key in self.hist:
            tf.summary.histogram(key, outputs[i], step=epoch + 1)
            i += 1
        for key in self.scalar:
            tf.summary.scalar(key, outputs[i], step=epoch + 1)
            i += 1
        for key in self.image:
            tf.summary.image(key, outputs[i], step=epoch + 1)
            i += 1


def str2bool(obj):
    if isinstance(obj, str):
        if obj == 'True':
            return True
        elif obj == 'False':
            return False
        else:
            raise TypeError('Type not support:{}'.format(obj))
    if isinstance(obj, bool):
        return obj
    else:
        raise TypeError('{} is not str'.format(obj))







from typing import List

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Layer

from retinanet.anchor import gpu
from retinanet.anchor.generator import generate_anchors
from retinanet.utils.image import resize_images


class UpSample(Layer):
    """
    Up-sampling refers to any technique that change an image to higher resolution.
    So it could be deconvolution, resizing or etc..

    The UpSampling layer modifies the source image to be the same as the shape of the target image.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)
        return resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class PriorProbability(keras.initializers.Initializer):
    """
    Apply a prior probability to the bias of the last layer in the classification subnet.
    """

    def __init__(self, prior=0.01):
        self.prior = prior

    def get_config(self):
        return {
            'prior': self.prior
        }

    def __call__(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype) * -np.log((1 - self.prior) / self.prior)


class Anchor(keras.layers.Layer):
    def __init__(self,
                 size: int,
                 stride: int,
                 ratios: List[float] = (0.5, 1, 2),
                 scales: List[float] = (2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)),
                 debug_inputs=None,
                 *args,
                 **kwargs):
        self.size = size
        self.stride = stride
        self.ratios = np.array(ratios, dtype=K.floatx())
        self.scales = np.array(scales, dtype=K.floatx())
        self.debug_inputs = debug_inputs

        self.n_anchors = len(self.ratios) * len(self.scales)
        self.anchors = K.variable(generate_anchors(base_size=self.size,
                                                   ratios=self.ratios,
                                                   scales=self.scales))

        super(Anchor, self).__init__(*args, **kwargs)

    def call(self, pyramid_features, *args, **kwargs) -> tf.Tensor:

        feature_shape = K.shape(pyramid_features)

        anchors = gpu.generate_shifted_anchors(pyramid_feature_shape=feature_shape[1:3],
                                               stride=self.stride,
                                               anchors=self.anchors)
        anchors = K.expand_dims(anchors, axis=0)
        anchors = K.tile(anchors, (feature_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[1:3]) * self.n_anchors
            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        config = super(Anchor, self).get_config()
        config.update({
            'size': self.size,
            'stride': self.stride,
            'scales': self.scales.tolist(),
            'ratios': self.ratios.tolist(),
            'n_anchors': self.n_anchors
        })

        return config


class RegressBoxes(keras.layers.Layer):
    """
    Apply regressions to boxes
    """

    def __init__(self, mean=(0, 0, 0, 0), std=(0.2, 0.2, 0.2, 0.2), *args, **kwargs):
        """
        :param mean: the mean value of the regressions for normalization
        :param std:  the standard deviation value of the regression for normalization
        """
        super(RegressBoxes, self).__init__(*args, **kwargs)
        self.mean = np.array(mean)
        self.std = np.array(std)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return gpu.apply_regression_to_boxes(anchors, regression, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std': self.std.tolist()})

        return config


class ClipBoxes(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = K.cast(K.shape(image), K.floatx())

        x1 = tf.clip_by_value(boxes[:, :, 0], 0, shape[2])
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, shape[1])
        x2 = tf.clip_by_value(boxes[:, :, 2], 0, shape[2])
        y2 = tf.clip_by_value(boxes[:, :, 3], 0, shape[1])

        return K.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]

from typing import List

import keras
import keras.backend as K
import numpy as np
from keras.layers import Layer

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
                 scales: List[float] = (2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.))):
        self.size = size
        self.stride = stride
        self.ratios = np.array(ratios, dtype=K.floatx())
        self.scales = np.array(scales, dtype=K.floatx())
        self.n_anchors = len(self.ratios) * len(self.scales)


from typing import List

import keras
import numpy as np
import tensorflow as tf
import keras.backend as K

from retinanet.anchor import gpu
from retinanet.anchor.generator import generate_anchors


class Anchors(keras.layers.Layer):
    """ Keras layer for generating achors for a given shape.
    """

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

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        feature_shape = K.shape(inputs)

        anchors = gpu.generate_shifted_anchors(pyramid_feature_shape=feature_shape[1:3],
                                               stride=self.stride,
                                               anchors=self.anchors)
        anchors = K.expand_dims(anchors, axis=0)
        anchors = K.tile(anchors, (feature_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[1:3]) * self.num_anchors
            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size': self.size,
            'stride': self.stride,
            'scales': self.scales.tolist(),
            'ratios': self.ratios.tolist(),
            'n_anchors': self.n_anchors
        })

        return config


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5,
                                                                                                        dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5,
                                                                                                        dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(
        keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors

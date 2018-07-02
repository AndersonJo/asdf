import keras.backend as K
import tensorflow as tf
import numpy as np


def generate_shifted_anchors(pyramid_feature_shape: tf.Tensor, stride: int, anchors: tf.Variable,
                             debug_inputs: tf.Tensor = None) -> tf.Tensor:
    """
    Generate shifted anchors based on the shape of the map and strike size
    :param pyramid_feature_shape: (height/stride, width/stride)
    :param stride: ...
    :param anchors: shape is (scales * ratios, 4). (min_x, min_y, max_x, max_y)
    :param debug_inputs: only used for debugging
    """
    height = pyramid_feature_shape[0]  # original image height / stride
    width = pyramid_feature_shape[1]  # original image width / stride

    # shift_x = [2, 6, 10, 14, ..., 594, 598] if width  = 600
    # shift_y = [2, 6, 10, 14, ..., 794, 798] if height = 800
    # shift_x = (arange(0,  width) + 0.5) * stride
    # shift_y = (arange(0, height) + 0.5) * stride
    shift_y = (K.arange(0, height, dtype=K.floatx()) + K.constant(0.5, dtype=K.floatx())) * stride
    shift_x = (K.arange(0, width, dtype=K.floatx()) + K.constant(0.5, dtype=K.floatx())) * stride

    # if debug_inputs is not None:
    #     sess = K.get_session()
    #     image = np.random.rand(1, 800, 600, 3)
    #     debug_y, debug_x = sess.run([shift_y, shift_x], feed_dict={debug_inputs: image})

    # shift_x
    # [[  2.,   6.,  10., ..., 590., 594., 598.],
    #  [  2.,   6.,  10., ..., 590., 594., 598.],
    #  ...
    #  [  2.,   6.,  10., ..., 590., 594., 598.]]
    #
    # shift_y
    # [[2, 2, 2, ..., 2, 2, 2],
    #  [6, 6, 6, ..., 6, 6, 6],
    #  ...
    #  [798, 798, 798, ..., 798, 798, 798]]
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

    # shift_x : [2, 6, 10, ..., 590, 594, 598]  shape: (30000, )
    # shift_y : [2, 2, 2, ... , 798, 798, 798]  shape: (30000, )
    shift_x = K.reshape(shift_x, [-1])
    shift_y = K.reshape(shift_y, [-1])

    # shifts shape: (4, 30000)
    # [[  2.,   6.,  10., ..., 590., 594., 598.],
    #  [  2.,   2.,   2., ..., 798., 798., 798.],
    #  [  2.,   6.,  10., ..., 590., 594., 598.],
    #  [  2.,   2.,   2., ..., 798., 798., 798.]]
    shifts = K.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    # shifts shape: (30000, 4)
    # [[  2.,   2.,   2.,   2.],
    #  [  6.,   2.,   6.,   2.],
    #  [ 10.,   2.,  10.,   2.],
    #  ...,
    #  [590., 798., 590., 798.],
    #  [594., 798., 594., 798.],
    #  [598., 798., 598., 798.]]
    shifts = K.transpose(shifts)

    n_anchors = K.shape(anchors)[0]  # 9 ... the number of anchors = scales * ratios
    n_points = K.shape(shifts)[0]  # 30000 number of anchors on the image = width/stride * height/stirde = 800/4 * 600*4

    # shape: (30000, 9, 4)
    shifted_anchors = K.reshape(anchors, [1, n_anchors, 4]) + K.cast(K.reshape(shifts, [n_points, 1, 4]), K.floatx())
    shifted_anchors = K.reshape(shifted_anchors, [n_points * n_anchors, 4])  # (270000, 4)

    return shifted_anchors


def apply_regression_to_boxes(boxes, deltas, mean=None, std=None):
    """
    Apply regression values (offset values between ground-truth anchors and anchors) to bounding boxes
    :param boxes: (Batch, n_boxes, 4). 4 -> (x1, y1, x2, y2)
    :param deltas: (d_x1, d_y1, d_x2, d_y2)
    :param mean: normalization mean value for regressions
    :param std: normalization standard deviation value for regressions
    :return: <nd.array> regression-applied boxes
    """

    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    regression_applied_boxes = K.stack([x1, y1, x2, y2], axis=2)

    return regression_applied_boxes


class RegressBoxes(object):
    pass

import keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image


def load_image(path: str, rgb=False) -> np.ndarray:
    """
    Load image.
    BGR is default
    """
    image = np.asarray(Image.open(path).convert('RGB'))
    if rgb:
        return image
    else:
        return image[:, :, ::-1].copy()


def normalize_image(image: np.ndarray):
    """
    Refers to https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
    image: BGR image
    """
    x = image.astype(K.floatx())
    data_format = K.image_data_format()
    mean = (103.939, 116.779, 123.68)

    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
    return x


def denormalize_image(image: np.ndarray):
    x = image.astype(K.floatx())
    data_format = K.image_data_format()
    mean = (103.939, 116.779, 123.68)

    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    return x


def flip_channel(image: np.ndarray):
    x = image.astype(K.floatx())
    data_format = K.image_data_format()

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
    return x


def resize_images(images, size, method='bilinear', align_corners=False):
    """
    Refer to https://www.tensorflow.org/versions/master/api_docs/python/tf/image/resize_images
    """
    methods = {
        'bilinear': tf.image.ResizeMethod.BILINEAR,
        'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic': tf.image.ResizeMethod.BICUBIC,
        'area': tf.image.ResizeMethod.AREA,
    }
    return tf.image.resize_images(images, size, method=methods[method], align_corners=align_corners)

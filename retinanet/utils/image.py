from PIL import Image
import numpy as np

import keras.backend as K


def image_load(path: str, rgb=False) -> np.ndarray:
    """
    Load image.
    BGR is default
    """
    image = np.asarray(Image.open(path).convert('RGB'))
    if rgb:
        return image
    else:
        return image[:, :, ::-1].copy()


def normalize_image(image: np.ndarray, flip_ch=False):
    """
    Refers to https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
    image: BGR image
    """
    x = image.astype(K.floatx())
    data_format = K.image_data_format()
    mean = (103.939, 116.779, 123.68)

    if flip_ch:
        x = flip_channel(x)

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


def denormalize_image(image: np.ndarray, flip_ch=False):
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

    if flip_ch:
        x = flip_channel(x)
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

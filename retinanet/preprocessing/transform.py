"""
The code refers to
"""

from typing import List, Tuple

import cv2
import numpy as np
import keras.backend as K

_DEFAULT_RS = np.random.RandomState()


class RandomTransformGenerator(object):
    def __init__(self,
                 min_rotation: float = 0,
                 max_rotation: float = 0,
                 min_translation: Tuple[float, float] = (0, 0),
                 max_translation: Tuple[float, float] = (0, 0),
                 min_shear: float = 0,
                 max_shear: float = 0,
                 min_scaling: Tuple[float, float] = (1., 1.),
                 max_scaling: Tuple[float, float] = (1., 1.),
                 flip_x: float = 0.,
                 flip_y: float = 0.,
                 seed=None):
        self.rand = np.random.RandomState(seed)

        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.min_translation = min_translation
        self.max_translation = max_translation
        self.min_shear = min_shear
        self.max_shear = max_shear
        self.min_scaling = min_scaling
        self.max_scaling = max_scaling
        self.flip_x = flip_x
        self.flip_y = flip_y

    def __next__(self):
        return np.linalg.multi_dot([
            rotate(self.min_rotation, self.max_rotation, self.rand),
            translate(self.min_translation, self.max_translation, self.rand),
            shear(self.min_shear, self.max_shear, self.rand),
            scale(self.min_scaling, self.max_scaling, self.rand),
            flip(self.flip_x, self.flip_y, self.rand)
        ])

    def __iter__(self):
        return self


def translation_matrix(translation):
    """
    Create a homogeneous translation matrix
    :param translation: a vector of translation factors
    :return: a translation 3 x 3 matrix
    """
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])


def scaling_matrix(factor):
    """
    Create a homogeneous scaling matrix
    :param factor: a vector for x and y
    :return: a scaled homogeneous 3 x 3 matrix
    """
    return np.array([
        [factor[0], 0, 0],
        [0, factor[1], 0],
        [0, 0, 1]
    ])


def rotating_matrix(angle):
    """
    Create a homogeneous rotated matrix
    :param angle: the angle in radians
    :return: a rotated 3 x 3 matrix
    """
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])


def shear_matrix(angle):
    """ Construct a homogeneous 2D shear matrix.
    # Arguments
        angle: the shear angle in radians
    # Returns
        the shear matrix as 3 by 3 numpy array
    """
    return np.array([
        [1, -np.sin(angle), 0],
        [0, np.cos(angle), 0],
        [0, 0, 1]
    ])


def _random_uniform_vector(min, max, rand=_DEFAULT_RS):
    """
    :param min: a minimum value for the uniform probability
    :param max: a maximum value for the uniform probability
    :param rand: a random generator
    :return: a vector of uniform probabilities
    """
    min = np.array(min)
    max = np.array(max)
    assert min.shape == max.shape
    assert len(min.shape) == 1
    return rand.uniform(min, max)


def rotate(min_radian, max_radian, rand=_DEFAULT_RS):
    """
    Creaate a random rotation matrix
    :param min_radian: a scalar for the minimum absolute angle in radians
    :param max_radian: a scalar for the maximum absolute angle in radians
    :param rand: a random generator
    :return: a homogeneous 3 x 3 rotation matrix
    """
    return rotating_matrix(rand.uniform(min_radian, max_radian))


def translate(min, max, rand=_DEFAULT_RS):
    """
    Create a random translation between min and max
    :param min: a vector of the minimum translation factors
    :param max: a vector of the maximum translation factors
    :param rand: a random generator
    :return: a homogeneous 3 x 3 translation matrix
    """
    return translation_matrix(_random_uniform_vector(min, max, rand))


def shear(min_radian, max_radian, rand=_DEFAULT_RS):
    """
    Create a random shear matrix
    :param min_radian: the minimum shear angle in radians
    :param max_radian: the maximum shear angle in radians
    :param rand: a random generator
    :return: a homogeneous 3 x 3 shear matrix
    """
    return shear_matrix(rand.uniform(min_radian, max_radian))


def scale(min_scaling, max_scaling, rand):
    """
    :param min_scaling: a vector of minimum scaling factors for x and y
    :param max_scaling: a vector of maximum scaling factors for x and y
    :param rand: random generator
    :return: a homogeneous 3 x 3 scaling matrix
    """
    return scaling_matrix(_random_uniform_vector(min_scaling, max_scaling, rand))


def flip(flip_x_chance, flip_y_chance, rand):
    """
    :param flip_x_chance: the probability to flip the image along the x axis
    :param flip_y_chance: the probability to flip the image along the y axis
    :param rand: a random generator
    :return: a homogeneous 3 x 3 scaled matrix
    """
    flip_x = rand.uniform(0, 1) < flip_x_chance
    flip_y = rand.uniform(0, 1) < flip_y_chance
    # 1 - 2 * bool gives 1 for False and -1 for True.
    return scaling_matrix((1 - 2 * flip_x, 1 - 2 * flip_y))


def change_origin(transform, center):
    """
    Adjust the origin of the transformation matrix by mo
    :param transform: the transformation matrix
    :param center: the new origin of the transformation
    :return: a new transformation matrix with the adjusted origin
    """
    center = np.array(center)

    return np.linalg.multi_dot([translation_matrix(center), transform, translation_matrix(-center)])


def adjust_transformation_for_image(transform, image, relative_translation):
    """
    Adjust the image to translation matrix so as to fit the size of the image to the translation matrix.
    The origin of the translation matrix will be centered.
    """
    height, width, channels = image.shape
    transform = transform.copy()

    # Scale the translation with the image size if specified.
    if relative_translation:
        transform[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    new_transformation = change_origin(transform, (0.5 * width, 0.5 * height))

    return new_transformation


def warp_affine(matrix: np.ndarray,
                image: np.ndarray,
                interpolation: int = cv2.INTER_LINEAR,
                border_mode: int = cv2.BORDER_REPLICATE,
                border_value: int = 0,
                channel_axis: int = 2) -> np.ndarray:
    """
    Apply the transformation matrix to the image

    :param matrix: a 3 x 3 homogeneous transformation matrix
    :param image: The image to apply transformation matrix
    :param interpolation: cv2 interpolation value (example cv2.INTER_??)
    :param border_value: cv2 border value
    :param border_mode: cv2 border mode (example cv2.BORDER_??)
    :param channel_axis: channel location in axis
    """

    if channel_axis != 2:
        image = np.moveaxis(image, channel_axis, 2)

    transformed_image = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize=(image.shape[1], image.shape[0]),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )

    if channel_axis != 2:
        transformed_image = np.moveaxis(transformed_image, 2, channel_axis)

    return transformed_image


def transform_bounding_box(transform: np.ndarray, mxmy: np.ndarray):
    """
    Transform the transformation matrix to an axis aligned bounding box
    :param transform: the transformation matrix to apply
    :param mxmy: (min_x, min_y, max_x, max_y)
    :return: a newly transformed mxmy vector (min_x, min_y, max_x, max_y)
    """
    x1, y1, x2, y2 = mxmy
    # Transform all 4 corners of the AABB.
    points = transform.dot([
        [x1, x2, x1, x2],
        [y1, y2, y2, y1],
        [1, 1, 1, 1],
    ])

    # Extract the min and max corners again.
    min_corner = points.min(axis=1)
    max_corner = points.max(axis=1)

    return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]


def rescale_image(image: np.ndarray, min_side: int = 800, max_side: int = 1333) -> Tuple[np.ndarray, float]:
    (height, width, _) = image.shape

    smallest_side = min(height, width)
    largest_side = max(height, width)

    # Calculate scaling ratio such that the smallest side is the min_side
    scale_ratio = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    if largest_side * scale_ratio > max_side:
        scale_ratio = max_side / largest_side

    # resize the image with the computed scale
    image = cv2.resize(image, None, fx=scale_ratio, fy=scale_ratio)

    return image, scale_ratio

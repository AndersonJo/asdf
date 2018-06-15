"""
The code refers to
"""

from typing import List, Tuple

import numpy as np

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

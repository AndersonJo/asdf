import keras
from PIL import Image

from retinanet.utils.image import image_load, normalize_image, denormalize_image
import os

from tests import DATASET_ROOT_PATH
import numpy as np
from scipy.misc import imsave


class TestImageTransform(object):
    def test_moveaxis(self):
        """
        이미지 채널을 첫번째로 옮기는 일처럼, array의 axis를 옮긴다.
        """

        image_path = os.path.join(DATASET_ROOT_PATH, 'VOC2007', 'JPEGImages', '000001.jpg')
        image = image_load(image_path, rgb=True)
        np.testing.assert_equal((500, 353, 3), image.shape)

        # Move channel to the first
        moved_image = np.moveaxis(image, -1, 0)
        np.testing.assert_equal((3, 500, 353), moved_image.shape)

        moved_image = np.moveaxis(image, 2, 0)
        np.testing.assert_equal((3, 500, 353), moved_image.shape)

        # Move channel to the second
        moved_image = np.moveaxis(image, -1, 1)
        np.testing.assert_equal((500, 3, 353), moved_image.shape)

        moved_image = np.moveaxis(image, 2, 1)
        np.testing.assert_equal((500, 3, 353), moved_image.shape)

    def test_normalize_image(self):
        image_path = os.path.join(DATASET_ROOT_PATH, 'VOC2007', 'JPEGImages', '000001.jpg')

        # Test1 RGB Image to normalize
        image = image_load(image_path, rgb=True)
        normalized_image = normalize_image(image)
        denormalized_image = denormalize_image(normalized_image)
        np.testing.assert_equal(image, denormalized_image)

        # Test2 BGR Image
        image = image_load(image_path, rgb=False)
        normalized_image = normalize_image(image)
        denormalized_image = denormalize_image(normalized_image)
        np.testing.assert_equal(image, denormalized_image)

        # Test3 BGR -> flip to RGB -> normalize -> denormalize -> flip to BGR -> BGR
        image = image_load(image_path, rgb=False)
        normalized_image = normalize_image(image, flip_ch=True)
        denormalized_image = denormalize_image(normalized_image, flip_ch=True)
        np.testing.assert_equal(image, denormalized_image)

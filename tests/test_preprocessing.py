import numpy as np

from retinanet.preprocessing.base import ImageGenerator
from retinanet.preprocessing.transform import RandomTransformGenerator


class TestBoundingBoxGenerator(object):
    def test_filter_bounding_box(self):
        """
        부적합한 bounding box를 제대로 걸러내는지 테스트한다
        """
        original_image_batch = [np.zeros((600, 600, 3))]

        # Test 1 : min_x and min_y are greater than max_x and max_y
        box_batch = [np.array([
            [5, 0, 25, 30],
            [150, 150, 50, 50],
            [150, 20, 50, 50],
            [10, 150, 50, 50]
        ])]
        expected_box_batch = [np.array([
            [5, 0, 25, 30],
        ])]

        filter_bounding_box = ImageGenerator.filter_bounding_box

        image_batch, box_batch = filter_bounding_box(original_image_batch, box_batch)
        np.testing.assert_equal(expected_box_batch, box_batch)
        np.testing.assert_equal(original_image_batch, image_batch)

        # Test2 : negative values
        box_batch = [np.array([
            [-1, 0, 10, 10],
            [0, -1, 10, 10],
            [0, 0, -1, 10],
            [0, 0, 10, -1],
            [-10, -10, -2, -1],
            [150, 150, 200, 200]
        ])]
        expected_box_batch = [np.array([
            [150, 150, 200, 200],
        ])]

        image_batch, box_batch = filter_bounding_box(original_image_batch, box_batch)
        np.testing.assert_equal(expected_box_batch, box_batch)
        np.testing.assert_equal(original_image_batch, image_batch)

        # Test3 : same values
        box_batch = [np.array([
            [5, 0, 25, 30],
            [0, 600, 200, 600],
            [200, 100, 200, 600]
        ])]
        expected_box_batch = [np.array([
            [5, 0, 25, 30],
        ])]

        image_batch, box_batch = filter_bounding_box(original_image_batch, box_batch)
        np.testing.assert_equal(expected_box_batch, box_batch)
        np.testing.assert_equal(original_image_batch, image_batch)


class TestTransformImage(object):
    def test_tranformation(self):
        rand = RandomTransformGenerator(
            min_rotation=0,
            max_rotation=np.pi,
            min_translation=(0, 0),
            max_translation=(1, 1),
            min_shear=0,
            max_shear=np.pi,
            min_scaling=(0, 0),
            max_scaling=(10, 10),
            flip_x=0.5,
            flip_y=0.5,
            seed=123)

        expected_matrix = np.array([[-4.164, 2.97, -0.3506],
                                    [5.867, -3.012, 0.10205],
                                    [0., 0., 1.]], dtype=np.float16)

        output = next(rand).astype(np.float16)
        np.testing.assert_equal(expected_matrix, output)

    def test_transform_mxmy(self):
        x1, y1, x2, y2 = (5, 10, 50, 40)
        transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        points = transform.dot([
            [x1, x2, x1, x2],
            [y1, y2, y2, y1],
            [1, 1, 1, 1],
        ])

        min_corner = points.min(axis=1)
        max_corner = points.max(axis=1)

        print(points)
        print(min_corner)
        print(max_corner)

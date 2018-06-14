import numpy as np

from retinanet.preprocessing.base import BoundingBoxGenerator


class TestBoundingBoxGenerator(object):
    def test_filter_bounding_box(self):
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

        generator = BoundingBoxGenerator(batch=2)
        image_batch, box_batch = generator.filter_bounding_box(original_image_batch, box_batch)
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

        image_batch, box_batch = generator.filter_bounding_box(original_image_batch, box_batch)
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

        image_batch, box_batch = generator.filter_bounding_box(original_image_batch, box_batch)
        np.testing.assert_equal(expected_box_batch, box_batch)
        np.testing.assert_equal(original_image_batch, image_batch)


class TestTransformImage(object):
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

from retinanet.preprocessing.pascal import PascalVOCGenerator
from retinanet.training.anchor import generate_targets, generate_anchors, shift, anchors_for_shape
import numpy as np


def test_generate_anchor():
    ratios = np.array([0.5, 1, 2])
    scales = np.array([1, 1.2, 1.5])
    anchors = generate_anchors(32, ratios=ratios, scales=scales)

    expected_anchors = [[-22.625, -11.3125, 22.625, 11.3125],
                        [-27.15625, -13.578125, 27.15625, 13.578125],
                        [-33.9375, -16.96875, 33.9375, 16.96875],
                        [-16.0, -16.0, 16.0, 16.0],
                        [-19.203125, -19.203125, 19.203125, 19.203125],
                        [-24.0, -24.0, 24.0, 24.0],
                        [-11.3125, -22.625, 11.3125, 22.625],
                        [-13.578125, -27.15625, 13.578125, 27.15625],
                        [-16.96875, -33.9375, 16.96875, 33.9375]]

    np.testing.assert_equal(expected_anchors, anchors.astype(np.float16))


def test_anchors_for_shape():
    anchors_for_shape([600, 800])

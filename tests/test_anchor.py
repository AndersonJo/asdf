import numpy as np

from retinanet.anchor.generator import generate_anchors, compute_overlap


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


def test_compute_overlap():
    # Test 1 : Overlaps
    anchors = np.array([[5, 3, 10, 10],
                        [0, 0, 5, 5],
                        [20, 20, 100, 100],
                        [100, 100, 200, 200]])
    gta = np.array([[5, 3, 10, 10],
                    [0, 0, 5, 5],
                    [20, 20, 50, 50]])

    expected_overlaps = [[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.140625],
                         [0.0, 0.0, 0.0]]

    overlaps = compute_overlap(anchors, gta)
    np.testing.assert_equal(expected_overlaps, overlaps)

    # Test 2 : Max overlaps
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    expected_max_overlaps = [1.0, 1.0, 0.140625, 0.0]
    np.testing.assert_equal(expected_max_overlaps, max_overlaps)

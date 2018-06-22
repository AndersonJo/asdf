import numpy as np


def test_np_minimum():
    # Test 1
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([5, 4, 1, 1, 7])

    expected = np.array([1, 2, 1, 1, 5])
    output = np.minimum(a, b)
    np.testing.assert_equal(expected, output)

    # Test 2
    # Broadcasting 을 이용하며, 처음 1이 [4, 2, 3] 하고 비교, 두번째 2가 [4, 2, 3] 하고 비교.. 계속 이런식
    a = np.array([[1], [2], [3], [4], [5]])
    b = np.array([4, 2, 3])

    expected = [[1, 1, 1],
                [2, 2, 2],
                [3, 2, 3],
                [4, 2, 3],
                [4, 2, 3]]
    output = np.minimum(a, b)
    np.testing.assert_equal(expected, output)

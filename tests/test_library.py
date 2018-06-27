import numpy as np
import keras.backend as K
import tensorflow as tf
import keras
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


class TestTensorFlow(object):
    def test_where(self):
        inputs = keras.layers.Input(shape=(3,))
        output = tf.where(tf.equal(inputs, 1))

        # Test where function
        sess = K.get_session()
        indicies = sess.run(output, feed_dict={inputs: [[1, 0, 0], [0, 1, 1]]})
        expected_output = [[0, 0], [1, 1], [1, 2]]
        np.testing.assert_equal(expected_output, indicies)

    def test_gather_nd(self):
        # Test gather_nd
        # where function으로 나온 indices를 사용해서 값을 꺼낼수 있다.
        sess = K.get_session()

        indices = np.array([[0., 0.], [1., 0.], [2., 1.]], dtype=np.int32)
        params = [['a', 'b'], ['c', 'd'], ['e', 'f']]

        indices_inputs = keras.Input((2,), dtype=tf.int32)
        params_inputs = keras.Input((2,), dtype=tf.string)

        pred = sess.run(tf.gather_nd(params_inputs, indices_inputs), feed_dict={params_inputs: params,
                                                                                indices_inputs: indices})
        expected_output = [b'a', b'c', b'f']
        np.testing.assert_equal(expected_output, pred)

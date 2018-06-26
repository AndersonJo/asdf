from tensorflow import set_random_seed

from retinanet.retinanet.model import RetinaNet
import keras.backend as K
import numpy as np


class TestSubNetwork(object):
    def test_prior_probability(self):
        retinanet = RetinaNet('resnet50', n_class=20, n_anchor=9)
        clf_model = retinanet.create_classification_subnet(20)

        sess = K.get_session()
        weights_with_prior_prob = sess.run(clf_model.weights[9])
        expected_prio_prob = np.array(-4.59512, dtype=np.float32)
        np.testing.assert_equal(expected_prio_prob, weights_with_prior_prob.mean())

    def test_regression_subnet(self):
        np.random.seed(0)
        set_random_seed(0)

        retinanet = RetinaNet('resnet50', n_class=5, n_anchor=9)
        reg_model = retinanet.create_regression_subnet(reg_feature_size=256)

        inputs = np.random.rand(1, 32, 32, 256)
        outputs = reg_model.predict(inputs)

        expected_mean = np.array(0.000437442, dtype=np.float32)
        np.testing.assert_equal(expected_mean, outputs.mean())


class TestRetinaNet(object):
    def test_create_retinanet(self):
        retinanet = RetinaNet('resnet50', n_class=10, n_anchor=9)
        retinanet.create_retinanet()

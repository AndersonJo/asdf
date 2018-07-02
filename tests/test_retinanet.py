from tensorflow import set_random_seed

from retinanet.retinanet.model import TrainingRetinaNet, RetinaNet
import keras.backend as K
import numpy as np


class TestSubNetwork(object):
    def test_prior_probability(self):
        retinanet = TrainingRetinaNet('resnet50', n_class=20)
        clf_model = retinanet.create_classification_subnet(20)

        sess = K.get_session()
        weights_with_prior_prob = sess.run(clf_model.weights[9])
        expected_prio_prob = np.array(-4.5951204, dtype=np.float32)
        np.testing.assert_equal(expected_prio_prob, weights_with_prior_prob.mean())

    def test_regression_subnet(self):
        np.random.seed(0)
        set_random_seed(0)

        retinanet = TrainingRetinaNet('resnet50', n_class=5)
        reg_model = retinanet.create_regression_subnet(reg_feature_size=256)

        inputs = np.random.rand(1, 32, 32, 256)
        outputs = reg_model.predict(inputs)
        expected_mean = np.array(-0.0006411766, dtype=np.float32)
        np.testing.assert_equal(expected_mean, outputs.mean())


class TestTrainingRetinaNet(object):
    def test_create_retinanet(self):
        retinanet = TrainingRetinaNet('resnet50', n_class=10)
        retinanet_model = retinanet.create_retinanet()

        images = np.random.rand(1, 800, 600, 3)
        clf_pred, reg_pred = retinanet_model.predict_on_batch(images)

        assert 10 == clf_pred.shape[-1]
        assert 4 == reg_pred.shape[-1]


class TestPredictionRetinaNet(object):

    def test_create_retinanet(self):
        retinanet = RetinaNet('resnet50', n_class=15)
        retinanet_model = retinanet.create_retinanet()

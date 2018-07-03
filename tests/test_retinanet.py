from tensorflow import set_random_seed

from retinanet.preprocessing.generator import create_data_generator
from retinanet.retinanet.losses import FocalLoss, SmoothL1Loss
from retinanet.retinanet.model import RetinaNet
import keras.backend as K
import numpy as np

from tests import DATASET_ROOT_PATH


class TestSubNetwork(object):
    def test_prior_probability(self):
        retinanet = RetinaNet('resnet50', n_class=20)
        clf_model = retinanet.create_classification_subnet(20)

        sess = K.get_session()
        weights_with_prior_prob = sess.run(clf_model.weights[9])
        expected_prio_prob = np.array(-4.5951204, dtype=np.float32)
        np.testing.assert_equal(expected_prio_prob, weights_with_prior_prob.mean())

    def test_regression_subnet(self):
        np.random.seed(0)
        set_random_seed(0)

        retinanet = RetinaNet('resnet50', n_class=5)
        reg_model = retinanet.create_regression_subnet(reg_feature_size=256)

        inputs = np.random.rand(1, 32, 32, 256)
        outputs = reg_model.predict(inputs)
        expected_mean = np.array(-0.0006411766, dtype=np.float32)
        np.testing.assert_equal(expected_mean, outputs.mean())


class TestTrainingRetinaNet(object):
    def test_create_retinanet(self):
        retinanet = RetinaNet('resnet50', n_class=10)
        model, training_model, prediction_model = retinanet()

        images = np.random.rand(1, 800, 600, 3)
        clf_pred, reg_pred = model.predict_on_batch(images)

        assert 10 == clf_pred.shape[-1]
        assert 4 == reg_pred.shape[-1]


class TestLoss(object):

    def test_loss(self):
        retinanet = RetinaNet('resnet50', n_class=15)
        model, training_model, prediction_model = retinanet()

        classes = {
            'bicycle': 1,
            'bird': 2,
            'bus': 5,
            'car': 6,
            'cat': 7,
            'chair': 8,
            'cow': 9,
            'diningtable': 10,
            'dog': 11,
            'horse': 12,
            'motorbike': 13,
            'person': 14,
            'sofa': 17,
            'train': 18,
            'tvmonitor': 19
        }

        train_generator, test_generator = create_data_generator('pascal', DATASET_ROOT_PATH, classes=classes)
        images, (clf_target, reg_target) = train_generator[0]
        clf_inputs, reg_inputs = training_model.targets

        loss = retinanet.focal_loss.info['loss']
        sess = K.get_session()

        def get(tensor):
            return sess.run(tensor, feed_dict={retinanet.inputs: images,
                                               clf_inputs: clf_target,
                                               reg_inputs: reg_target})

        assert isinstance(get(loss), np.float32)

    def test_loss_names(self):
        focal_loss = FocalLoss()
        l1_loss = SmoothL1Loss()

        assert 'FocalLoss' == str(focal_loss)
        assert 'FocalLoss' == focal_loss.__name__
        assert 'SmoothL1Loss' == str(l1_loss)
        assert 'SmoothL1Loss' == l1_loss.__name__

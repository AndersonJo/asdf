import cv2
import numpy as np

from retinanet.preprocessing.generator import create_data_generator
from retinanet.preprocessing.pascal import PascalVOCGenerator
from retinanet.retinanet.model import RetinaNet
from retinanet.utils.image import denormalize_image
from tests import DATASET_ROOT_PATH


class TestEvaluation(object):
    def test_all_annotations(self):
        np.random.seed(0)
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

        retinanet = RetinaNet('resnet50')

        model, model_train, model_pred = retinanet(n_class=len(classes))
        images = np.random.rand(1, 800, 600, 3)

        train_generator, test_generator = create_data_generator('pascal', DATASET_ROOT_PATH, batch=2, classes=classes)
        limit = 10

        evaluate(retinanet, test_generator)

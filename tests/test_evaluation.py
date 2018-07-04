from keras import Model
import numpy as np

from retinanet.preprocessing.generator import create_data_generator
from retinanet.preprocessing.pascal import PascalVOCGenerator
from retinanet.retinanet.model import RetinaNet
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

        retinanet = RetinaNet('resnet50', n_class=20)
        model, model_train, model_pred = retinanet()
        images = np.random.rand(1, 800, 600, 3)

        train_generator, test_generator = create_data_generator('pascal', DATASET_ROOT_PATH, batch=2, classes=classes)
        limit = 10

        evaluate(retinanet, test_generator)


def evaluate(retinanet: RetinaNet,
             generator: PascalVOCGenerator,
             score_threshold: float = 0.05,
             max_detections: int = 300,
             limit: int = 10):
    """
    :param retinanet: Retinanet Instance
    :param generator: Validation generator
    :param score_threshold: The score confidence threshold
    :param max_detections: The maximum number of detections per image
    :param limit: Limit the evaluating data size
    :return:
    """
    for i, idx in enumerate(range(generator.size())):
        if i >= limit:
            break

        # Get image data
        batch = generator.get_batch(idx)
        image_batch, boxes_true, scales = generator.load_batch(batch)
        image_batch = generator.process_inputs(image_batch)

        # Predict
        boxes, scores, labels = retinanet.predict_on_batch(image_batch, scales)
        scores = np.random.rand(generator.batch_size, 300) - 0.8

        # Select indices which have scores above the threshold
        indices = np.where(scores > score_threshold)

        # Select by indices
        boxes = boxes[indices]
        scores = scores[indices]
        labels = labels[indices]

        sorted_score_indices = np.argsort(-scores)[:max_detections]
        indices = indices[0][sorted_score_indices]
        sorted_boxes = boxes[sorted_score_indices]
        sorted_scores = scores[sorted_score_indices]
        sorted_labels = labels[sorted_score_indices]

        draw_boxes()



        import ipdb
        ipdb.set_trace()
    pass

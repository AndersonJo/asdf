from retinanet.preprocessing.sinsinsa import SinsinsaGenerator, SINSINSA_CLASSES
from tests import DATASET_ROOT_PATH
import numpy as np


def test_sinsinsa_generator():
    generator = SinsinsaGenerator(DATASET_ROOT_PATH, voc_challenges=['Sinsinsa2018'], classes=SINSINSA_CLASSES)
    assert ['Sinsinsa2018'] == generator.voc_challenges

    selected = np.random.choice(range(1, generator.size() + 1), 10)
    for i in selected:
        image, (clf_target, reg_target) = generator[i]

        batch, height, width, color = image.shape
        assert 1 == batch
        assert 800 <= height
        assert 800 <= width
        assert 3 == color

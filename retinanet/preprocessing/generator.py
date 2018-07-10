import argparse

from retinanet.preprocessing.pascal import PascalVOCGenerator, VOC_CLASSES
from retinanet.preprocessing.sinsinsa import SinsinsaGenerator
from retinanet.preprocessing.transform import RandomTransformGenerator


def create_data_generator(data_mode: str, data_path: str, batch: int = 1, classes: dict = VOC_CLASSES,
                          random_transform: bool = False):
    mode = data_mode.lower().strip()
    data_path = data_path.strip()

    if random_transform:
        random_generator = RandomTransformGenerator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x=0.5,
            flip_y=0.5,
            seed=123)
    else:
        random_generator = RandomTransformGenerator(
            flip_x=0.5,
            flip_y=0.5,
            seed=123)

    if mode == 'pascal':
        train_generator = PascalVOCGenerator(data_path, voc_mode='train', batch=batch,
                                             random_generator=random_generator, classes=classes)
        test_generator = PascalVOCGenerator(data_path, voc_mode='test', batch=batch,
                                            random_generator=random_generator, classes=classes)
    elif mode == 'sinsinsa':
        train_generator = SinsinsaGenerator(data_path, voc_mode='train', batch=batch,
                                            voc_challenges=['Sinsinsa2018'],
                                            random_generator=random_generator, classes=classes)
        test_generator = SinsinsaGenerator(data_path, voc_mode='test', batch=batch,
                                           voc_challenges=['Sinsinsa2018'],
                                           random_generator=random_generator, classes=classes)
    else:
        raise ValueError('Invalid data generator {0} received'.format(mode))
    return train_generator, test_generator

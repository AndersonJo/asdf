import os
import sys

# Enable relative import

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))

import argparse
import keras.backend as K
import tensorflow as tf

from retinanet.retinanet.model import RetinaNet
from retinanet.preprocessing.pascal import PascalVOCGenerator
from retinanet.preprocessing.transform import RandomTransformGenerator


def parse_args(args):
    parser = argparse.ArgumentParser(description='Retinanet training script')
    parser.add_argument('data_mode')
    parser.add_argument('data_path', default='/data/VOCdevkit/')
    parser.add_argument('--steps', type=int, default=10000, help='number of steps per epoch')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')

    # Backbone
    parser.add_argument('--backbone', default='resnet101', type=str, help='Backbone model (resnet50)')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone layers when training')
    parser.add_argument('--weights', default=None, type=str,
                        help='weights path for backbone modle. (default is pre-trained ImageNet')

    # Sub Networks
    parser.add_argument('--clf-feature', default=256, type=int, help='The feature size of classification sub-network')
    parser.add_argument('--reg-feature', default=256, type=int, help='The feature size of classification sub-network')

    parser.add_argument('--batch', default=1, type=int, help='Batch size')

    parser.add_argument('--gpu', type=str, help='GPU ID to use')
    parser.add_argument('--random-transform', help='Randomly transform image and boxes', action='store_true')

    # Model
    parser.add_argument('--checkpoint', type=str, help='Start training from the checkpoint')
    return parser.parse_args(args)


def define_gpu(gpu: str):
    """
    What GPU to use (check nvidia-smi)
    :param gpu: GPU ID
    """
    if gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def set_session(sess=None) -> tf.Session:
    if sess is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    K.set_session(sess)
    return sess


def create_data_generator(parser: argparse.ArgumentParser):
    mode = parser.data_mode.lower().strip()
    data_path = parser.data_path.strip()

    if parser.random_transform:
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
        train_generator = PascalVOCGenerator(data_path, voc_mode='train', random_generator=random_generator)
        test_generator = PascalVOCGenerator(data_path, voc_mode='test', random_generator=random_generator)
    else:
        raise ValueError('Invalid data generator {0} received'.format(mode))
    return train_generator, test_generator


def train():
    parser = parse_args(sys.argv[1:])

    # Specify which GPU to use
    if parser.gpu:
        define_gpu(parser.gpu)

    # Set Session
    set_session()

    # Create Generator
    train_generator, test_generator = create_data_generator(parser)

    # Create RetinaNet
    retinanet = RetinaNet(parser.backbone, n_class=20)
    training_model, pred_model = retinanet.create_retinanet(freeze_backbone=parser.freeze_backbone,
                                                            weights=parser.weights,
                                                            clf_feature_size=parser.clf_feature,
                                                            reg_feature_size=parser.reg_feature,
                                                            prior_probability=0.01)

    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=parser.steps,
        epochs=parser.epochs,
        verbose=1
    )


if __name__ == '__main__':
    train()

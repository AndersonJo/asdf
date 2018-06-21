import argparse
import os
import sys

# Enable relative import
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))

import keras.backend as K
import tensorflow as tf

from retinanet.backbone import load_backbone


def parse_args(args):
    parser = argparse.ArgumentParser(description='Retinanet training script')
    parser.add_argument('data_mode')
    parser.add_argument('data-path', default='/data/VOCdevkit/VOC2012/')
    parser.add_argument('--backbone', default='resnet50', type=str, help='Backbone model (resnet50)')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone layers when training')
    parser.add_argument('--batch', default=1, type=int, help='Batch size')

    parser.add_argument('--gpu', type=str, help='GPU ID to use')
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


def train():
    parser = parse_args(sys.argv[1:])

    # Specify which GPU to use
    if parser.gpu:
        define_gpu(parser.gpu)

    # Set Session
    set_session()

    # Load Backbone
    backbone = load_backbone(parser.backbone)


if __name__ == '__main__':
    train()

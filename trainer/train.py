import argparse
import os
import sys

# Enable relative import
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))

from typing import List

import keras.backend as K
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, ReduceLROnPlateau

from retinanet.callbacks.eval import Evaluate
from retinanet.callbacks.wrapper import ModelWrapperCallback
from retinanet.preprocessing.generator import create_data_generator
from retinanet.retinanet.model import RetinaNet


def parse_args(args):
    parser = argparse.ArgumentParser(description='Retinanet training script')
    parser.add_argument('data_mode')
    parser.add_argument('data_path', default='/data/VOCdevkit/')
    parser.add_argument('--steps', type=int, default=10000, help='number of steps per epoch')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')

    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help='Backbone model (resnet50)')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze backbone layers when training')
    parser.add_argument('--weights', default=None, type=str,
                        help='weights path for backbone modle. (default is pre-trained ImageNet')

    # Sub Networks
    parser.add_argument('--clf-feature', default=256, type=int, help='The feature size of classification sub-network')
    parser.add_argument('--reg-feature', default=256, type=int, help='The feature size of classification sub-network')

    parser.add_argument('--batch', default=1, type=int, help='Batch size')

    parser.add_argument('--gpu', type=str, default="0", help='GPU ID to use')
    parser.add_argument('--random-transform', help='Randomly transform image and boxes', action='store_true')

    # Model
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Start training from the checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file name to load the model')

    # TensorBoard
    parser.add_argument('--tensorboard', type=str, default='/tmp/retinanet-anderson-tensorboard',
                        help='the path of tensorboard directory')

    return parser.parse_args(args)


def define_gpu(gpu: str):
    """
    What GPU to use (check nvidia-smi)
    :param gpu: GPU ID
    """
    if gpu:
        gpu = str(gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def set_session(sess=None) -> tf.Session:
    if sess is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    K.set_session(sess)
    return sess


def create_callbacks(backbone: str,
                     data_mode: str,
                     batch_size: int,
                     validation_generator,
                     prediction_model,
                     checkpoint_path: str = 'checkpoints',
                     tf_board_dir: str = '/tmp/retinanet-anderson-tensorboard') -> List[Callback]:
    # Init
    backbone = backbone.lower().strip()
    data_mode = data_mode.lower().strip()

    callbacks = []
    # TensorBoard Callback
    if tf_board_dir:
        tensorboard_callback = TensorBoard(
            log_dir=tf_board_dir,
            histogram_freq=0,
            batch_size=batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    # Save Model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    save_path = os.path.join(checkpoint_path,
                             '{backbone}_{data_mode}_{{epoch:02d}}.h5'.format(
                                 backbone=backbone, data_mode=data_mode))

    checkpoint = ModelCheckpoint(
        save_path,
        verbose=1,
        # save_best_only=True,
        # monitor="mAP",
        # mode='max'
    )
    callbacks.append(checkpoint)

    # Evaluation Callback
    if validation_generator:
        evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback)
        evaluation = ModelWrapperCallback(evaluation, prediction_model)
        callbacks.append(evaluation)

    # Decaying learning rate
    callbacks.append(ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='auto',

        cooldown=0,
        min_lr=0
    ))
    return callbacks


def train():
    parser = parse_args(sys.argv[1:])

    # Specify which GPU to use
    if parser.gpu:
        define_gpu(parser.gpu)

    # Set Session
    set_session()

    # Create Generator
    train_generator, validation_generator = create_data_generator(data_mode=parser.data_mode,
                                                                  data_path=parser.data_path,
                                                                  random_transform=parser.random_transform)

    # Create RetinaNet
    retina_kwargs = dict(
        freeze_backbone=parser.freeze_backbone,
        weights=parser.weights,
        clf_feature_size=parser.clf_feature,
        reg_feature_size=parser.reg_feature,
        prior_probability=0.01
    )
    retinanet = RetinaNet(parser.backbone, n_class=20)
    model, training_model, pred_model = retinanet(**retina_kwargs)

    # Create callbacks
    callbacks = create_callbacks(backbone=parser.backbone,
                                 data_mode=parser.data_mode,
                                 batch_size=parser.batch,
                                 validation_generator=validation_generator,
                                 prediction_model=pred_model,
                                 checkpoint_path=parser.checkpoint_dir,
                                 tf_board_dir=parser.tensorboard)

    # Training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=parser.steps,
        epochs=parser.epochs,
        callbacks=callbacks,
        verbose=1)


if __name__ == '__main__':
    train()

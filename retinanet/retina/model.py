import keras
from keras import Input, Model

from retinanet.backbone import load_backbone
from retinanet.retina.layer import PriorProbability


class RetinaNet(object):
    def __init__(self,
                 backbone: str,
                 n_class: int,
                 n_anchor: int = 9,
                 checkpoint: str = None,
                 weights: str = None,
                 freeze: bool = False,

                 fpn_size=256,
                 ):
        """
        # Basic Parameters
        :param n_class: the number of classes
        :param n_anchor: the number of anchors

        # Backbone Parameters
        :param backbone: The name of the backbone Model
        :param checkpoint: Checkpoint file path. It resume training from the checkpoint.
        :param freeze: freeze the weights of the backbone model
        :param weights: weights file path. if None, it uses pre-trained ImageNet weights.

        # Pyramid Parameters
        :param fpn_size: the feature size of the pyramid network
        """
        # Set basic parameters
        self.n_class = n_class
        self.n_anchor = n_anchor

        # Set Feature Pyramid Network

        # Set Backbone Network
        self.inputs = Input(shape=(None, None, 3), name='input')
        self.backbone = load_backbone(backbone)

        # Load snapshot to resume training from the snapshot
        if checkpoint is not None:
            model = keras.models.load_model(checkpoint, custom_objects=self.backbone.custom_objects)
        else:
            # Default weights are pre-trained ImageNet weights
            if weights is None:
                weights = self.backbone.download_imagenet()

            backbone_model = self.backbone.create_backbone_model(self.inputs, freeze=freeze)
            # TODO: Retina + FPN
            self.create_classification_subnet()

            backbone_model.load_weights(weights, by_name=True, skip_mismatch=False)

    def create_regression_subnet(self):
        pass

    def create_classification_subnet(self,
                                     n_class: int,
                                     n_anchor: int = 9,
                                     clf_feature_size: int = 256,
                                     prior_prob: float = 0.01,
                                     name='clf_subnet_model') -> Model:

        def clf_conv1(h, name):
            return keras.layers.Conv2D(
                filters=clf_feature_size,
                activation='relu',
                name=name,
                kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros',
                kernel_size=3,
                strides=1,
                padding='same'
            )(h)

        def clf_conv2(h, name):
            return keras.layers.Conv2D(
                filters=n_class * n_anchor,
                kernel_initializer=keras.initializers.zeros(),
                bias_initializer=PriorProbability(prior=prior_prob),
                name=name,
                kernel_size=3,
                strides=1,
                padding='same')(h)

        inputs = Input(shape=(None, None, self))

        h = clf_conv1(inputs, 'clf_subnet_1')
        h = clf_conv1(h, 'clf_subnet_2')
        h = clf_conv1(h, 'clf_subnet_3')
        h = clf_conv1(h, 'clf_subnet_4')
        h = clf_conv2(h, 'clf_subnet_5_with_prior_bias')

        h = keras.layers.Reshape((-1, n_class), name='clf_subnet_reshape')(h)
        h = keras.layers.Activation('sigmoid', name='clf_subnet_sigmoid')(h)
        return Model(inputs=inputs, outputs=h, name=name)

from typing import List

import keras
import keras.backend as K
import tensorflow as tf
from keras import Model
from keras.layers import Input

from retinanet.backbone import load_backbone
from retinanet.retinanet.layers import PriorProbability
from retinanet.retinanet.pyramid import graph_pyramid_features, apply_pyramid_features
from retinanet.utils.anchor_information import AnchorInfo


class TrainingRetinaNet(object):
    def __init__(self,
                 backbone: str,
                 n_class: int,
                 n_anchor: int = 9,
                 anchor_info: AnchorInfo = AnchorInfo(),
                 fpn_feature_size: int = 256):
        """
        # Basic Parameters
        :param n_class: the number of classes
        :param n_anchor: the number of anchors
        :param anchor_info: AnchorInfo

        # Backbone Parameters
        :param backbone: The name of the backbone Model

        :param freeze: freeze the weights of the backbone model
        :param weights: weights file path. if None, it uses pre-trained ImageNet weights.

        # Pyramid Parameters
        :param fpn_feature_size: the feature size of the pyramid network
        """
        # Initialize basic parameters
        self.n_class = n_class
        self.n_anchor = n_anchor
        self.anchor_info = anchor_info

        # Initialize Feature Pyramid Network
        self.fpn_feature_size = fpn_feature_size

        # Initialize Backbone Network
        self.inputs = Input(shape=(None, None, 3), name='input')
        self.backbone = load_backbone(backbone)

        # Initialize RetinaNet
        self._retinanet = None

    @property
    def model(self) -> Model:
        return self._retinanet

    def create_retinanet(self,
                         # Backbone
                         freeze_backbone: bool = False,
                         weights: str = None,

                         # Sub Networks
                         clf_feature_size: int = 256,
                         reg_feature_size: int = 256,
                         prior_probability=0.01) -> Model:
        # Initialize Backbone Model
        backbone_model = self.backbone.create_backbone_model(self.inputs, freeze=freeze_backbone)

        # Create Sub Networks
        clf_subnet = self.create_classification_subnet(clf_feature_size=clf_feature_size,
                                                       prior_prob=prior_probability)

        reg_subnet = self.create_regression_subnet(reg_feature_size=reg_feature_size)

        # Apply Feature Pyramid
        pyramid_features = graph_pyramid_features(*backbone_model.outputs)
        pyramids = apply_pyramid_features(pyramid_features, clf_subnet, reg_subnet)

        # Create RetinaNet Model
        self._retinanet = Model(inputs=self.inputs, outputs=pyramids, name='retinanet')

        # Load weights
        if weights is None:
            weights = self.backbone.download_imagenet()
        self._retinanet.load_weights(weights, by_name=True, skip_mismatch=False)

        self._retinanet.compile(
            loss={'regression': self.smooth_l1_loss(),
                  'classification': self.focal_loss()},
            optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001))

        return self.model

    def create_regression_subnet(self, reg_feature_size: int = 256) -> Model:
        """
        :param reg_feature_size: Regression subnet's feature size

        Every layers are the same as the ones in classification sub network except the final one.
        All layers in subnets are initialized with bias b = 0 and a Guassian weight fill with stddev = 0.01
        :return: regression subnet model
        """
        n_anchor = self.n_anchor
        fpn_feature_size = self.fpn_feature_size

        def regr_conv1(h, name):
            return keras.layers.Conv2D(
                filters=reg_feature_size,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu',
                kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros',
                name=name,
            )(h)

        def regr_conv2(h, name):
            return keras.layers.Conv2D(
                filters=n_anchor * 4,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
                bias_initializer='zeros',
                name=name,
            )(h)

        inputs = Input(shape=(None, None, fpn_feature_size), name='reg_subnet_input')
        h = regr_conv1(inputs, 'reg_subnet_1')
        h = regr_conv1(h, 'reg_subnet_2')
        h = regr_conv1(h, 'reg_subnet_3')
        h = regr_conv1(h, 'reg_subnet_4')
        h = regr_conv2(h, 'reg_subnet_5')

        h = keras.layers.Reshape((-1, 4), name='reg_subnet_reshape')(h)
        h = keras.layers.Activation('linear', name='reg_subnet_linear_activation')(h)
        return keras.models.Model(inputs=inputs, outputs=h, name='reg_subnet_model')

    def create_classification_subnet(self,
                                     clf_feature_size: int = 256,
                                     prior_prob: float = 0.01) -> Model:
        """
        :param clf_feature_size: Classification subnet's feature size
        :param prior_prob:
        :param name:
        :return:
        """
        n_class = self.n_class
        n_anchor = self.n_anchor
        fpn_feature_size = self.fpn_feature_size

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

        inputs = Input(shape=(None, None, fpn_feature_size), name='clf_subnet_input')

        h = clf_conv1(inputs, 'clf_subnet_1')
        h = clf_conv1(h, 'clf_subnet_2')
        h = clf_conv1(h, 'clf_subnet_3')
        h = clf_conv1(h, 'clf_subnet_4')
        h = clf_conv2(h, 'clf_subnet_5_with_prior_bias')

        h = keras.layers.Reshape((-1, n_class), name='clf_subnet_reshape')(h)
        h = keras.layers.Activation('sigmoid', name='clf_subnet_sigmoid')(h)
        return Model(inputs=inputs, outputs=h, name='clf_subnet_model')

    def smooth_l1_loss(self, sigma=3.0):
        """
        Create a smooth L1 loss function
        :param sigma: the point where the loss changes from L2 to L1
        :return: Smooth L1 function
        """
        sigma_squared = sigma ** 2

        def _smooth_l1(y_true, y_pred):
            """
            :param y_true: (batch, n_proposals, 5) -> the last value is the state of the anchor (positive, negative, ignore)
                                                    positive -> 1, negative -> 0, ignore  -> -1
            :param y_pred: (batch, n_proposals, 4)
            """
            # Separate target and state
            regr_true = y_true[:, :, :4]
            anchor_state = y_true[:, :, 4]

            # Only use positive anchors
            # You do not have to perform regression on useless backgrounds
            positive_indices = tf.where(K.equal(anchor_state, 1))
            regression_pred = tf.gather_nd(y_pred, positive_indices)
            regression_true = tf.gather_nd(regr_true, positive_indices)

            # Smooth L1 Loss
            # x = abs(y_pred - y_true)
            #
            # f(x) = 0.5 * sigma^2 * (x)^2      if |x| < 1 / sigma^2
            #        x - 0.5 / sigma^2          otherwise
            x = regression_pred - regression_true
            x = K.abs(x)

            regression_loss = tf.where(
                K.less(x, 1.0 / sigma_squared),
                0.5 * sigma_squared * K.pow(x, 2),
                x - 0.5 / sigma_squared)

            # compute the normalizer: the number of positive anchors
            normalizer = K.maximum(1, K.shape(positive_indices)[0])
            normalizer = K.cast(normalizer, dtype=K.floatx())
            return K.sum(regression_loss) / normalizer

        return _smooth_l1

    def focal_loss(self, alpha=0.25, gamma=2.0):
        """
        Create a focal loss function
        """

        def _focal(y_true, y_pred):
            """
            Focal Loss Function
            :param y_true: (batch, n_proposals, n_classes)
            :param y_pred: (batch, n_proposals, n_classes
            :return: Focal loss
            """
            labels = y_true
            classification = y_pred

            # filter out "ignore" anchors
            anchor_state = K.max(y_true, axis=2)
            indices = tf.where(K.not_equal(anchor_state, -1))  # -1 is ignore state. indices = {object, background}
            labels = tf.gather_nd(labels, indices)
            classification = tf.gather_nd(classification, indices)

            # compute the focal loss
            # alpha_factor = alpha      if object
            #                1-alpha    if background
            # For example
            #       object alpha        = 0.25
            #       background alpha    = 0.75
            alpha_factor = K.ones_like(labels) * alpha
            alpha_factor = tf.where(K.equal(labels, 1), alpha_factor, 1 - alpha_factor)

            # focal_weight
            # focal_weight = 1 - pred   if object
            #                pred       if background
            focal_weight = tf.where(K.equal(labels, 1), 1 - classification, classification)
            focal_weight = alpha_factor * focal_weight ** gamma

            cls_loss = focal_weight * K.binary_crossentropy(labels, classification)

            # compute the normalizer: the number of positive anchors
            normalizer = tf.where(K.equal(anchor_state, 1))
            normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
            normalizer = K.maximum(1.0, normalizer)

            return K.sum(cls_loss) / normalizer

        return _focal

    def load_checkpoint(self, checkpoint: str) -> Model:
        """
        :param checkpoint: Checkpoint file path. It resume training from the checkpoint.
        :return: RetinaNet Model
        """
        model = keras.models.load_model(checkpoint, custom_objects=self.backbone.custom_objects)
        return model


class RetinaNet(TrainingRetinaNet):
    def create_retinanet(self,
                         # Backbone
                         freeze_backbone: bool = False,
                         weights: str = None,
                         pyramids: List[str] = ('P2', 'P3', 'P4', 'P5', 'P6', 'P7'),

                         # Sub Networks
                         clf_feature_size: int = 256,
                         reg_feature_size: int = 256,
                         prior_probability=0.01,

                         ) -> Model:
        super(RetinaNet, self).create_retinanet(freeze_backbone=freeze_backbone,
                                                weights=weights,
                                                clf_feature_size=clf_feature_size,
                                                reg_feature_size=reg_feature_size,
                                                prior_probability=prior_probability)

        # Get Pyramid Features
        pyramids = list(map(lambda p: p.lower(), pyramids))
        pyramid_features = [self.model.get_layer(p_name) for p_name in pyramids]

        self.generate_anchors(pyramid_features)

    def generate_anchors(self, pyramid_features: List[tf.Tensor]):
        pass

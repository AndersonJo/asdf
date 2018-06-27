import keras
from keras import Model
from keras.layers import Input

from retinanet.backbone import load_backbone
from retinanet.retinanet import losses
from retinanet.retinanet.layer import PriorProbability
from retinanet.retinanet.pyramid import graph_pyramid_features, apply_pyramid_features


class RetinaNet(object):
    def __init__(self,
                 backbone: str,
                 n_class: int,
                 n_anchor: int = 9,

                 fpn_feature_size: int = 256,
                 ):
        """
        # Basic Parameters
        :param n_class: the number of classes
        :param n_anchor: the number of anchors

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

        # TODO: Compile the model
        self._retinanet.compile(
            loss={'regression': losses.smooth_l1(),
                  'classification': losses.focal()},
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

    def load_checkpoint(self, checkpoint: str) -> Model:
        """
        :param checkpoint: Checkpoint file path. It resume training from the checkpoint.
        :return: RetinaNet Model
        """
        model = keras.models.load_model(checkpoint, custom_objects=self.backbone.custom_objects)
        return model

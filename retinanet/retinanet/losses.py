import tensorflow as tf
import keras.backend as K


class BaseLoss(object):
    def __init__(self):
        self._info = dict()
        self.__name__ = str(self)

    @property
    def info(self) -> dict:
        return self._info

    def add_info(self, key: str, value: tf.Tensor):
        self._info[key] = value


class SmoothL1Loss(BaseLoss):

    def __init__(self, sigma: float = 3.0):
        """
        Create a smooth L1 loss function
        :param sigma: the point where the loss changes from L2 to L1
        """
        super(SmoothL1Loss, self).__init__()
        self.sigma = sigma
        self.sigma_squared = sigma ** 2

    def __str__(self):
        return 'SmoothL1Loss'

    def __call__(self, y_true, y_pred):
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
            K.less(x, 1.0 / self.sigma_squared),
            0.5 * self.sigma_squared * K.pow(x, 2),
            x - 0.5 / self.sigma_squared)

        # compute the normalizer: the number of positive anchors
        normalizer = K.maximum(1, K.shape(positive_indices)[0])
        normalizer = K.cast(normalizer, dtype=K.floatx())
        return K.sum(regression_loss) / normalizer


class FocalLoss(BaseLoss):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __str__(self):
        return 'FocalLoss'

    def __call__(self, y_true, y_pred):
        """
        Focal Loss Function
        :param y_true: (batch, n_proposals, n_classes)
        :param y_pred: (batch, n_proposals, n_classes
        :return: Focal loss
        """
        labels = y_true
        classification = y_pred

        self.add_info('y_true', y_true)
        self.add_info('y_pred', y_pred)

        # filter out "ignore" anchors
        anchor_state = K.max(y_true, axis=2)
        indices = tf.where(K.not_equal(anchor_state, -1))  # -1 is ignore state. indices = {object, background}
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        self.add_info('anchor_state', anchor_state)
        self.add_info('indices', indices)
        self.add_info('labels', labels)
        self.add_info('classification', classification)

        # compute the focal loss
        # alpha_factor = alpha      if object
        #                1-alpha    if background
        # For example
        #       object alpha        = 0.25
        #       background alpha    = 0.75
        alpha_factor = K.ones_like(labels) * self.alpha
        alpha_factor = tf.where(K.equal(labels, 1), alpha_factor, 1 - alpha_factor)

        # focal_weight
        # focal_weight = 1 - pred   if object
        #                pred       if background
        focal_weight = tf.where(K.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** self.gamma

        cls_loss = focal_weight * K.binary_crossentropy(labels, classification)

        self.add_info('alpha_factor', alpha_factor)
        self.add_info('focal_weight', focal_weight)
        self.add_info('focal_loss', cls_loss)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(K.equal(anchor_state, 1))
        normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer = K.maximum(1.0, normalizer)

        loss = K.sum(cls_loss) / normalizer

        self.add_info('normalizer', normalizer)
        self.add_info('loss', loss)

        return loss

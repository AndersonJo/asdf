import tensorflow as tf
import keras.backend as K


def smooth_l1_loss(sigma=3.0):
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


def focal_loss(alpha: object = 0.25, gamma: object = 2.0) -> object:
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

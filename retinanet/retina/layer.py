import math

import keras
import numpy as np


class PriorProbability(keras.initializers.Initializer):
    """
    Apply a prior probability to the bias of the last layer in the classification subnet.
    """

    def __init__(self, prior=0.01):
        self.prior = prior

    def get_config(self):
        return {
            'prior': self.prior
        }

    def __call__(self, shape, dtype=None):
        result = np.ones(shape, dtype=dtype) * -math.log((1 - self.prior) / self.prior)

        return result

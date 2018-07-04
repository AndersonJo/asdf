from typing import List
import numpy as np
import keras.backend as K


class AnchorInfo(object):
    def __init__(self, sizes: List[int] = (16, 32, 64, 128, 256, 512),
                 strides: List[int] = (4, 8, 16, 32, 64, 128),
                 ratios: List[float] = (0.5, 1, 2),
                 scales: List[float] = (2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.))):
        """
        :param sizes: a list of sizes. Each size corresponds to the one feature level
        :param strides: a list of strides. Each stride corresponds to the one feature level
        :param ratios: a list of ratios per location in a feature map
        :param scales: a list of scales per location in a feature map
        """
        self.sizes = sizes
        self.strides = strides
        self.ratios = np.array(ratios, dtype=K.floatx())
        self.scales = np.array(scales, dtype=K.floatx())
        self._n_anchor = None

    def count_anchors(self) -> int:
        if self._n_anchor is None:
            self._n_anchor = len(self.ratios) * len(self.scales)
        return self._n_anchor

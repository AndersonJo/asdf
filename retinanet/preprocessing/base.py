import warnings
from abc import ABC, abstractmethod
from typing import List, Union, Tuple

import numpy as np
from keras.utils import Sequence


class BaseGenerator(Sequence, ABC):

    def __init__(self, batch: int = 1, shuffle: bool = True):
        self.batch_size = batch
        self.shuffle = shuffle
        self._data = list()
        self.filters = list()

    @abstractmethod
    def load_data(self, *args, **kwargs):
        raise NotImplementedError('load_data method not implemented')

    def __getitem__(self, index):
        batch = self._data[index * self.batch_size: (index + 1) * self.batch_size]
        return batch

    def __len__(self):
        """
        Number of batch in the Sequence
        """
        return int(np.ceil(len(self._data) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            pass


class BoundingBoxGenerator(BaseGenerator):

    def __init__(self, batch: int = 1, shuffle: bool = True):
        super(BoundingBoxGenerator, self).__init__(batch=batch, shuffle=shuffle)

    def __getitem__(self, index):
        batch = super(BoundingBoxGenerator, self).__getitem__(index)
        self.filter_bounding_box(batch)

    def load_data(self, data: List[np.ndarray]):
        self._data = data

    @staticmethod
    def filter_bounding_box(image_batch: List[np.ndarray], box_batch: List[np.ndarray]) -> Tuple[
        List[np.ndarray], List[np.ndarray]]:
        for i, (image, boxes) in enumerate(zip(image_batch, box_batch)):
            invalid_indices = np.where(
                (boxes[:, 2] <= boxes[:, 0]) |
                (boxes[:, 3] <= boxes[:, 1]) |
                (boxes[:, 0] < 0) |
                (boxes[:, 1] < 0) |
                (boxes[:, 2] > image.shape[1]) |
                (boxes[:, 3] > image.shape[0])
            )[0]

            if len(invalid_indices):
                filtered_boxes = np.delete(boxes, invalid_indices, axis=0)
                box_batch[i] = filtered_boxes

        return image_batch, box_batch

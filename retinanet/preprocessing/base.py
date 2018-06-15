from abc import ABC
from typing import List, Tuple

import numpy as np
from keras.utils import Sequence

from retinanet.preprocessing.transform import RandomTransformGenerator
from retinanet.utils.image import normalize_image


class BaseGenerator(Sequence, ABC):

    def __init__(self, batch: int = 1, shuffle: bool = True):
        self.batch_size = batch
        self.shuffle = shuffle
        self._data = list()
        self.filters = list()

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


class ImageGenerator(BaseGenerator):

    def __init__(self, batch: int = 1, shuffle: bool = True,
                 random_generator: RandomTransformGenerator = None):
        super(ImageGenerator, self).__init__(batch=batch, shuffle=shuffle)
        self.random_generator = random_generator

    def __getitem__(self, index):
        batch = super(ImageGenerator, self).__getitem__(index)
        box_batch = self.load_box_batch(batch)
        image_batch = self.load_image_batch(batch)

        # Filter invalid bounding boxes
        image_batch, box_batch = self.filter_bounding_box(image_batch, box_batch)

        # Perform image pre-processing
        self.preprocess_batch(image_batch, box_batch)

        return batch

    def load_image(self, image_index):
        raise NotImplementedError('load_image method not implemented')

    def load_box(self, box_index):
        raise NotImplementedError('load_box method not implemented')

    def load_image_batch(self, batch):
        return [self.load_image(image_index) for image_index in batch]

    def load_box_batch(self, batch):
        return [self.load_box(box_index) for box_index in batch]

    def preprocess_batch(self, image_batch, box_batch):
        for i, (image, boxes) in enumerate(zip(image_batch, box_batch)):
            # pre-process image
            image = self.preprocess_image(image)

            # Get translation
            image, boxes = self.perform_translation(image, boxes)
            # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def preprocess_image(self, image) -> np.ndarray:
        return normalize_image(image)

    def perform_translation(self, image, boxes):
        translation_matrix = next(self.random_generator)
        # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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

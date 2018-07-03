from abc import ABC
from typing import List, Tuple

import cv2
import keras.backend as K
import numpy as np
from keras.utils import Sequence

from retinanet.preprocessing.transform import RandomTransformGenerator, adjust_transformation_for_image, warp_affine, \
    transform_bounding_box, rescale_image
from retinanet.anchor.generator import generate_targets
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

    def _set_data(self, data):
        self._data = data

    def on_epoch_end(self):
        if self.shuffle:
            pass


class ImageGenerator(BaseGenerator):
    INTERPOLATION = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos4': cv2.INTER_LANCZOS4
    }

    BORDER_MODE = {
        'constant': cv2.BORDER_CONSTANT,
        'nearest': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'wrap': cv2.BORDER_WRAP

    }

    def __init__(self, batch: int = 1, shuffle: bool = True,
                 random_generator: RandomTransformGenerator = RandomTransformGenerator(),

                 relative_translation: bool = True,
                 interpolation: str = 'linear',
                 border: str = 'nearest',
                 border_value: int = 0,
                 channel_axis: int = 2,

                 image_min_size: int = 800,
                 image_max_size: int = 1600):
        super(ImageGenerator, self).__init__(batch=batch, shuffle=shuffle)
        self.random_generator = random_generator

        # Transformation parameters
        self.relative_translation = relative_translation
        self.interpolation = self.INTERPOLATION[interpolation]
        self.border_mode = self.BORDER_MODE[border]
        self.border_value = border_value

        # Set channel axis
        if K.image_data_format() == 'channels_last':
            self.channel_axis = 2
        elif K.image_data_format() == 'channels_first':
            self.channel_axis = 0
        else:
            self.channel_axis = channel_axis

        # Set image min and max size
        self.image_min_size = image_min_size
        self.image_max_size = image_max_size

    def __getitem__(self, index):
        image_batch, box_batch = self.get_batch(index)

        # Combine a list of images into a single nd-array of images
        inputs = self.process_inputs(image_batch)

        # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        targets = self.process_targets(image_batch, box_batch)

        return inputs, targets

    def get_batch(self, index: int) -> Tuple[list, list]:
        batch = super(ImageGenerator, self).__getitem__(index)
        box_batch = self.load_annotation_batch(batch)
        image_batch = self.load_image_batch(batch)

        # Filter invalid bounding boxes
        image_batch, box_batch = self.filter_invalid_bounding_box_batch(image_batch, box_batch)

        # Perform image pre-processing
        image_batch, box_batch = self.preprocess_batch(image_batch, box_batch)
        return image_batch, box_batch

    def count_class(self) -> int:
        raise NotImplementedError('count_class method not implemented')

    def name_to_label(self, name):
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        raise NotImplementedError('label_to_name method not implemented')

    def load_image(self, image_info):
        raise NotImplementedError('load_image method not implemented')

    def load_annotation(self, annotation_info):
        raise NotImplementedError('load_annotation method not implemented')

    def load_image_batch(self, batch):
        image_batch = [self.load_image(image_index) for image_index in batch]
        return list(filter(lambda x: x is not None, image_batch))

    def load_annotation_batch(self, batch):
        annotation_batch = [self.load_annotation(annotation_index) for annotation_index in batch]
        return list(filter(lambda x: x is not None, annotation_batch))

    def preprocess_batch(self, image_batch: list, box_batch: list) -> Tuple[list, list]:
        for i, (image, boxes) in enumerate(zip(image_batch, box_batch)):
            # pre-process image
            image = self.preprocess_image(image)

            # Get translation
            image, boxes = self.perform_translation(image, boxes)

            # Resize Image
            image, scale_ratio = self.resize_image(image)
            boxes[:, :4] *= scale_ratio

            image_batch[i] = image
            box_batch[i] = boxes
        return image_batch, box_batch

    def preprocess_image(self, image) -> np.ndarray:
        return normalize_image(image)

    def perform_translation(self, image: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # Adjust translation matrix for image
        transformation_matrix = adjust_transformation_for_image(next(self.random_generator), image,
                                                                self.relative_translation)

        # Perform cv2.warpAffine with the transformation matrix applied to the image
        image = warp_affine(transformation_matrix, image, self.interpolation, self.border_mode,
                            self.border_value, channel_axis=self.channel_axis)

        # Adjust the bounding boxes to fit transformation matrix
        for i in range(boxes.shape[0]):
            boxes[i, :4] = transform_bounding_box(transformation_matrix, boxes[i, :4])
        return image, boxes

    def resize_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        return rescale_image(image, self.image_min_size, self.image_max_size)

    def process_inputs(self, images: List[np.ndarray]):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in images) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=K.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(images):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def process_targets(self, image_batch, box_batch):
        return generate_targets(image_batch, box_batch, batch_size=self.batch_size, n_classes=self.count_class())

    @staticmethod
    def filter_invalid_bounding_box_batch(image_batch: List[np.ndarray], box_batch: List[np.ndarray]) -> Tuple[
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

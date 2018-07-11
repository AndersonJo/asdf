import cv2
import numpy as np
from keras import Model

from retinanet.preprocessing.pascal import PascalVOCGenerator
from retinanet.utils.colors import label_color
from retinanet.utils.image import denormalize_image


class Evaluator(object):

    def __init__(self, inference_model: Model, generator: PascalVOCGenerator, save_dir: str = 'temp'):
        self.model = inference_model
        self.generator = generator
        self.save_dir = save_dir

    def __call__(self, iou_threshold=0.5, score_threshold=0.05, max_detections=100, limit: int = None):
        if limit is None:
            limit = self.generator.size() + 1

        for i, idx in enumerate(range(self.generator.size())):
            if i >= limit:
                break

            # Get image data
            annotation_info = self.generator.get_single_data(idx)  # (data_dir, filename)
            annotation = self.generator.load_annotation(annotation_info)  # [(x1, y1, x2, y2, label), ...] for an image
            raw_image = self.generator.load_image(annotation_info)  # Raw BGR Image (without scale or transformation)

            # Filter invalid boxes
            annotation = self.generator.filter_invalid_bounding_box(raw_image, annotation)

            # Preprocess
            image = self.generator.preprocess_image(raw_image.copy())  # Normalize the image

            # Resize Image
            image, scale_ratio = self.generator.resize_image(image)

            # Get Detections
            detections = self.predict_detections(image, score_threshold, max_detections)

            # Denormalize
            image = denormalize_image(image)

            # self.draw_boxes(raw_image, annotation, color=(0, 0, 255), thickness=1)
            self.draw_detections(image, detections, detections[:, 4], detections[:, 5],
                                 thickness=1)

            cv2.imwrite('{}/haha{}.png'.format(self.save_dir, i), image)

    def predict_detections(self, image, score_threshold: float = 0.05, max_detections: int = 300, ) -> np.ndarray:
        """
        :param image: (height, width, 3) a single image
        :param scale: rescaling floating point value
        :param score_threshold: all predicted boxes less than score threshold will be dropped
        :param max_detections: the maximum number of detections to limit
        :return:  boxes with score and label. ((x1, y1, x2, y2, label, score), ...)
        """

        # Predict with the inference model
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))

        # Select indices which are over the threshold score
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # Sort scores in descending order. shape : (300, )
        sorted_scores = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes = boxes[0, indices[sorted_scores], :]
        image_scores = scores[sorted_scores]
        image_labels = labels[0, indices[sorted_scores]]  # (300, )

        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_labels, axis=1), np.expand_dims(image_scores, axis=1)], axis=1)

        return image_detections

    @classmethod
    def draw_detections(cls, image, boxes, labels, scores, color=None, thickness=2, score_threshold=0.5):
        selection = np.where(scores > score_threshold)[0]
        for i in selection:
            c = color if color is not None else label_color(int(labels[i]))

            cls.draw_box(image, boxes[i, :], color=c, thickness=thickness)

    @classmethod
    def draw_boxes(cls, image, boxes, color=(0, 255, 0), thickness=2):
        for box_idx in range(boxes.shape[0]):
            cls.draw_box(image, boxes[box_idx], color, thickness)

    @classmethod
    def draw_box(cls, image, box, color=(255, 0, 0), thickness=2):
        """
        :param image: the original image
        :param box: (x1, y1, x2, y2)
        :param color: RGB colors as a tuple
        :param thickness: ...
        """
        b = np.array(box).astype(int)
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

import cv2
import numpy as np
from keras import Model

from retinanet.preprocessing.pascal import PascalVOCGenerator
from retinanet.retinanet.inference import predict_on_batch
from retinanet.utils.image import denormalize_image


class Evaluator(object):

    def __init__(self, inference_model: Model, generator: PascalVOCGenerator, save_dir: str = 'tmp'):
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
            filtered_boxes = self.generator.filter_invalid_bounding_box(raw_image, annotation)

            # Preprocess
            image = self.generator.preprocess_image(raw_image.copy())  # Normalize the image

            # Resize Image
            image, scale_ratio = self.generator.resize_image(image)

            # Get Detections
            detections = self.get_detections(image, scale_ratio, score_threshold, max_detections)
            #
            # # Draw
            # denormalized_image = denormalize_image(raw_image)

            self.draw_boxes(raw_image, annotation, color=(0, 0, 255), thickness=1)
            self.draw_boxes(raw_image, detections, color=(255, 255, 0), thickness=1)

            cv2.imwrite('tmp/haha{}.png'.format(i), raw_image)

            import ipdb
            ipdb.set_trace()

    def get_detections(self, image, scale,
                       score_threshold: float = 0.05, max_detections: int = 300, ) -> np.ndarray:
        """
        :param image: (height, width, 3) a single image
        :param scale: rescaling floating point value
        :param score_threshold: all predicted boxes less than score threshold will be dropped
        :param max_detections: the maximum number of detections to limit
        :return:  boxes with score and label. ((x1, y1, x2, y2, label, score), ...)
        """

        # Predict with the inference model
        images, boxes, scores, labels = predict_on_batch(self.model,
                                                         np.expand_dims(image, axis=0),
                                                         scale)

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
    def draw_boxes(cls, image, boxes, color=(0, 255, 0), thickness=2):
        for box_idx in range(boxes.shape[0]):
            cls.draw_box(image, boxes[box_idx], color, thickness)

    @classmethod
    def draw_box(cls, image, box, color, thickness=2):
        """
        :param image: the original image
        :param box: (x1, y1, x2, y2)
        :param color: RGB colors as a tuple
        :param thickness: ...
        """
        b = np.array(box).astype(int)
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def evaluate22(inference_model: Model,
               generator: PascalVOCGenerator,
               score_threshold: float = 0.05,
               max_detections: int = 300,
               limit: int = 10):
    """
    :param retinanet: Retinanet Instance
    :param generator: Validation generator
    :param score_threshold: The score confidence threshold
    :param max_detections: The maximum number of detections per image
    :param limit: Limit the evaluating data size
    :return:
    """
    for i, idx in enumerate(range(generator.size())):
        if i >= limit:
            break

        # Get image data
        batch = generator.get_batch(idx)
        image_batch, boxes_true, scales = generator.load_batch(batch)
        image_batch = generator.process_inputs(image_batch)

        # get_detections(inference_model)

        # # scores = np.random.rand(generator.batch_size, 300) - 0.8
        #
        # # Select indices which have scores above the threshold
        # indices = np.where(scores > score_threshold)
        #
        # # Select by indices
        # boxes = boxes[indices]
        # scores = scores[indices]
        # labels = labels[indices]
        #
        # sorted_score_indices = np.argsort(-scores)[:max_detections]
        # indices = indices[0][sorted_score_indices]
        # sorted_boxes = boxes[sorted_score_indices]
        # sorted_scores = scores[sorted_score_indices]
        # sorted_labels = labels[sorted_score_indices]
        #
        # for image_idx in range(image_batch.shape[0]):
        #     image = image_batch[image_idx]
        #
        #     # Draw ground-truth boxes
        #     gtboxes = boxes_true[image_idx]
        #     draw_boxes(image, gtboxes)
        #
        #     import ipdb
        #     ipdb.set_trace()
        #     # Draw predicted boxes
        #     boxes = sorted_boxes[image_idx]
        #     draw_boxes(image, boxes, color=(255, 255, 0))
        #
        # # draw_annotations(image_batch, boxes_true, label_to_name=generator.label_to_name)
        #
        # cv2.imwrite('haha1.png', image_batch[0])
        # cv2.imwrite('haha2.png', image_batch[1])
        #
        # import ipdb
        # ipdb.set_trace()

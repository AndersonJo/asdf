import cv2

import numpy as np

from retinanet.preprocessing.pascal import PascalVOCGenerator
from retinanet.retinanet.model import RetinaNet
from retinanet.utils.image import denormalize_image


def evaluate(retinanet: RetinaNet,
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

        # Predict
        boxes, scores, labels = retinanet.predict_on_batch(image_batch, scales)
        image_batch = denormalize_image(image_batch)
        scores = np.random.rand(generator.batch_size, 300) - 0.8

        # Select indices which have scores above the threshold
        indices = np.where(scores > score_threshold)

        # Select by indices
        boxes = boxes[indices]
        scores = scores[indices]
        labels = labels[indices]

        sorted_score_indices = np.argsort(-scores)[:max_detections]
        indices = indices[0][sorted_score_indices]
        sorted_boxes = boxes[sorted_score_indices]
        sorted_scores = scores[sorted_score_indices]
        sorted_labels = labels[sorted_score_indices]

        for image_idx in range(image_batch.shape[0]):
            image = image_batch[image_idx]

            # Draw ground-truth boxes
            gtboxes = boxes_true[image_idx]
            draw_boxes(image, gtboxes)

            # Draw predicted boxes
            boxes = sorted_boxes[image_idx]
            draw_boxes(image, boxes, color=(255, 255, 0))

        # draw_annotations(image_batch, boxes_true, label_to_name=generator.label_to_name)

        cv2.imwrite('haha1.png', image_batch[0])
        cv2.imwrite('haha2.png', image_batch[1])

        import ipdb
        ipdb.set_trace()
    pass


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for box_idx in range(boxes.shape[0]):
        draw_box(image, boxes[box_idx], color, thickness)


def draw_box(image, box, color, thickness=2):
    """
    :param image: the original image
    :param box: (x1, y1, x2, y2)
    :param color: RGB colors as a tuple
    :param thickness: ...
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

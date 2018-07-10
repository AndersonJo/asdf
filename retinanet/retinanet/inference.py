import numpy as np

from retinanet.utils.image import denormalize_image


def predict_on_batch(inference_model, image_batch: np.ndarray, scales: list = None):
    boxes, scores, labels = inference_model.predict_on_batch(image_batch)

    # Get back image scales to the original size
    if scales is not None:
        boxes = (boxes.T / scales).T

    image_batch = denormalize_image(image_batch)
    return image_batch, boxes, scores, labels

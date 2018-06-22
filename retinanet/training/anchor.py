import numpy as np
import keras.backend as K


def generate_targets(image_batch: np.ndarray, box_batch: np.ndarray, batch_size: int):
    # get the max image shape
    max_shape = tuple(max(image.shape[x] for image in image_batch) for x in range(3))

    # compute labels and regression targets
    labels_group = [None] * batch_size
    regression_group = [None] * batch_size
    for index, (image, boxes) in enumerate(zip(image_batch, box_batch)):
        labels_group[index], boxes, anchors = anchor_targets_bbox(max_shape, boxes, 20,  # self.count_class(),
                                                                  mask_shape=image.shape)

        regression_group[index] = bbox_transform(anchors, boxes)
        import ipdb
        ipdb.set_trace()

        # append anchor states to regression targets (necessary for filtering 'ignore', 'positive' and 'negative' anchors)
        anchor_states = np.max(labels_group[index], axis=1, keepdims=True)
        regression_group[index] = np.append(regression_group[index], anchor_states, axis=1)

    labels_batch = np.zeros((batch_size,) + labels_group[0].shape, dtype=K.floatx())
    regression_batch = np.zeros((batch_size,) + regression_group[0].shape, dtype=K.floatx())

    # copy all labels and regression values to the batch blob
    for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
        labels_batch[index, ...] = labels
        regression_batch[index, ...] = regression

    return [regression_batch, labels_batch]


def anchor_targets_bbox(
        image_shape,
        boxes,
        num_classes,
        mask_shape=None,
        negative_overlap=0.4,
        positive_overlap=0.5,
        **kwargs):
    """

    :param image_shape: (height, width)
    :param boxes:
    :param num_classes:
    :param mask_shape:
    :param negative_overlap:
    :param positive_overlap:
    :param kwargs:
    :return:
    """
    anchors = anchors_for_shape(image_shape, **kwargs)

    # Each anchor should have at least one class vector.
    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.ones((anchors.shape[0], num_classes)) * -1

    if boxes.shape[0]:
        # obtain indices of gt boxes with the greatest overlap
        overlaps = compute_overlap(anchors, boxes[:, :4])

        argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < negative_overlap, :] = 0

        # compute box regression targets
        boxes = boxes[argmax_overlaps_inds]

        # fg label: above threshold IOU
        positive_indices = max_overlaps >= positive_overlap
        labels[positive_indices, :] = 0
        labels[positive_indices, boxes[positive_indices, 4].astype(int)] = 1
    else:
        # if there is no box then everything should be background.
        labels[:] = 0
        boxes = np.zeros_like(anchors)

    # ignore boxes outside of image
    mask_shape = image_shape if mask_shape is None else mask_shape
    anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
    indices = np.logical_or(anchors_centers[:, 0] >= mask_shape[1], anchors_centers[:, 1] >= mask_shape[0])
    labels[indices, :] = -1

    return labels, boxes, anchors


def layer_shapes(image_shape, model):
    """Compute layer shapes given input image shape and the model.

    :param image_shape:
    :param model:
    :return:
    """
    shape = {
        model.layers[0].name: (None,) + image_shape,
    }

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            inputs = [shape[lr.name] for lr in node.inbound_layers]
            if not inputs:
                continue
            shape[layer.name] = layer.compute_output_shape(inputs[0] if len(inputs) == 1 else inputs)

    return shape


def make_shapes_callback(model):
    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        image_shapes = [shape["P{}".format(level)][1:3] for level in pyramid_levels]
        return image_shapes

    return get_shapes


def guess_shapes(image_shape, pyramid_levels):
    """Guess shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
        image_shape,
        pyramid_levels=None,
        ratios=None,
        scales=None,
        strides=None,
        sizes=None,
        shapes_callback=None):
    """
    Generate all anchors for each pyramid levels.
    Each pyramid level has all anchors
    :param image_shape: (height, width)
    :return: All anchors for all pyramid levels. all_anchors = (None, 4)
    """
    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]
    if strides is None:
        strides = [2 ** x for x in pyramid_levels]
    if sizes is None:
        # FPN uses single scales { 32^2, 64^2, 128^2, 256^2, 512^2 }
        # sizes = [32, 64, 128, 256, 512]
        sizes = [2 ** (x + 2) for x in pyramid_levels]
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    """
    Generate anchors for an image.
    :param shape: (height, width)
    :param stride: stride value
    :param anchors: anchors (x1, y1, x2, y2)
    :return: all anchors (x1, y1, x2, y2) for an image.
    """
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # shifts are (None, 4) matrix.
    # each vector has (x, y, x, y) location of the image
    # (ex. shifts.shape = (7500, 4))
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(base_size: int = 16, ratios: np.ndarray = None, scales: np.ndarray = None):
    """
    Generate anchors; the center of the anchor is (0, 0)
    :return: anchors (x1, y1, x2, y2) ; the center point is (0, 0)
    """
    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # Initialize zero matrix
    anchors = np.zeros((num_anchors, 4))

    # scale base_size (0, 0, width, height)
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    # it is like creating multi-scale square boxes and calculate the area of the square boxes
    # 32 x 32, 38.4 x 38.4, 48 x 48 ... repeat
    # areas = [1024, 1474.56, 2304,
    #          1024, 1474.56, 2304,
    #          1024, 1474.56, 2304]
    areas = anchors[:, 2] * anchors[:, 3]

    # sqrt(Area / ratio) = the width of square box
    # (ex. 32 x 32 = 1024 -> sqrt(1024 / 1) = 32)
    # height = width * ratio
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    # x2 = width - width/2
    # y2 = height - height/2
    # x1 = 0 - width/2
    # y1 = 0 - height/2
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
    targets = targets.T

    targets = (targets - mean) / std

    return targets


def compute_overlap(a, b):
    """
    :param a: (N, 4) ndarray of float
    :param b: (K, 4) ndarray of float
    :return: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    # Replace negative values ( v < 0 ) with 0
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    intersection = iw * ih

    # Union area = Areas of a + Areas of b - intersection
    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - intersection
    ua = np.maximum(ua, np.finfo(float).eps)

    return intersection / ua

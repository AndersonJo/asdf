import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union, Dict
import numpy as np

from retinanet.preprocessing.base import ImageGenerator
from retinanet.utils.image import load_image

VOC_CLASSES = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}


class PascalVOCGenerator(ImageGenerator):
    # VOC2007 uses test.txt as test data but this generator simply uses val.txt for test
    # VOC2017 uses val.txt as test data
    TEST_FILES = ('val.txt',)
    TRAIN_FILES = ('trainval.txt',)
    CHALLENGES = ('VOC2007', 'VOC2012')

    def __init__(self,
                 voc_root_path: str = '/data/VOCdevkit',
                 voc_challenges: List[str] = CHALLENGES,
                 voc_mode: str = 'train',
                 classes: Dict[str, int] = VOC_CLASSES,
                 convert_classes: Dict[str, str] = None,
                 limit_size: int = None,
                 **kwargs):
        """
        :param voc_root_path: the path of "VOCdevkit" including VOC2007 or VOC2012 (i.e. '/data/VOCdevkit')
        :param voc_challenges: VOC challenge data (i.e. ('VOC2007', 'VOC20010', 'VOC2012'))
        :param voc_mode: 'train' or 'test'

        voc_root_path path should look like this
        =======================================
        ├── VOC2007
        │   ├── Annotations
        │   ├── ImageSets
        │   │   ├── Layout
        │   │   ├── Main
        │   │   └── Segmentation
        │   ├── JPEGImages
        │   ├── SegmentationClass
        │   └── SegmentationObject
        ├── VOC2012
        │   ├── Annotations
        │   ├── ImageSets
        │   │   ├── Action
        │   │   ├── Layout
        │   │   ├── Main
        │   │   └── Segmentation
        │   ├── JPEGImages
        │   ├── SegmentationClass
        │   └── SegmentationObject
        =======================================
        """
        super(PascalVOCGenerator, self).__init__(**kwargs)

        # Initialize VOC data
        self.voc_root_path = voc_root_path
        self.voc_challenges = voc_challenges
        self.voc_mode = voc_mode.lower().strip()
        self.limit_size = limit_size

        # Set classes and labels
        self.classes = classes
        self.convert_classes = convert_classes
        self.labels = {v: k for k, v in self.classes.items()}

        self.init_data()

    def init_data(self, limit_size=None) -> List[Tuple[str, str]]:
        # VOC data absolute paths
        # i.e. ['/data/VOCdevkit/VOC2007', '/data/VOCdevkit/VOC2012']
        challenge_paths = [os.path.join(self.voc_root_path, challenge_name) for challenge_name in self.voc_challenges]
        challenge_paths = list(filter(lambda p: os.path.exists(p), challenge_paths))

        if self.voc_mode == 'train':
            data_files = self.TRAIN_FILES
        else:
            data_files = self.TEST_FILES

        data = list()
        for challenge_path in challenge_paths:
            _dataset_paths = [os.path.join(challenge_path, 'ImageSets', 'Main', t) for t in data_files]
            _dataset_paths = list(filter(lambda t: os.path.exists(t), _dataset_paths))

            if len(_dataset_paths) <= 0:
                continue

            for dataset_path in _dataset_paths:
                with open(dataset_path) as f:
                    for line in f:
                        data.append((challenge_path, line.strip()))

        self._set_data(data)
        return data

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def count_class(self):
        return len(self.classes)

    def load_image(self, image_info):
        challenge_path, filename = image_info
        image_path = os.path.join(challenge_path, 'JPEGImages', '{0}.jpg'.format(filename))
        if not os.path.exists(image_path):
            image_path = os.path.join(challenge_path, 'JPEGImages', '{0}.png'.format(filename))
            if not os.path.exists(image_path):
                raise FileNotFoundError('{0} image file not found'.format(image_path))

        return load_image(image_path)

    def load_annotation(self, annotation_info):
        challenge_path, filename = annotation_info
        annotation_path = os.path.join(challenge_path, 'Annotations', '{0}.xml'.format(filename))
        # if not os.path.exists(annotation_path):
        #     raise FileNotFoundError('{0} annotation file not found'.format(annotation_path))

        try:
            tree = ET.parse(annotation_path)
            boxes = self.__parse_bounding_boxes(tree.getroot())
            if not len(boxes):
                return None
            return boxes

        except ET.ParseError as e:
            raise ValueError('invalid annotations file: {0}: {1}'.format(annotation_path, e))
        except ValueError as e:
            raise ValueError('invalid annotations file: {0}: {1}'.format(annotation_path, e))

    def parse_annotation(self, element) -> Union[None, np.ndarray]:
        class_name = _find_node(element, 'name').text
        if class_name not in self.classes:
            return None

        box = np.zeros((1, 5))
        box[0, 4] = self.name_to_label(class_name)

        bndbox = _find_node(element, 'bndbox')
        box[0, 0] = _find_node(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
        box[0, 1] = _find_node(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
        box[0, 2] = _find_node(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
        box[0, 3] = _find_node(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1
        return box

    def __parse_bounding_boxes(self, xml_root):
        size_node = _find_node(xml_root, 'size')
        # width = _find_node(size_node, 'width', 'size.width', parse=float)
        # height = _find_node(size_node, 'height', 'size.height', parse=float)

        boxes = np.zeros((0, 5))
        for i, element in enumerate(xml_root.iter('object')):
            box = self.parse_annotation(element)
            if box is None:
                continue

            boxes = np.append(boxes, box, axis=0)
        return boxes


def _find_node(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise ValueError('illegal value for \'{}\': {}'.format(debug_name, e))
    return result

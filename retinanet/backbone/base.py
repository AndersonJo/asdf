from abc import ABC, abstractmethod

from keras import Model, Input

from retinanet.retinanet import losses
from retinanet.retinanet.initializer import PriorProbability
from retinanet.retinanet import layers
from retinanet.utils.filter_detections import FilterDetections


class BackboneBase(ABC):

    def __init__(self):
        self.custom_objects = {
            'UpSample': layers.UpSample,
            'PriorProbability': PriorProbability,
            'RegressBoxes': layers.RegressBoxes,
            'FilterDetections': FilterDetections,
            'Anchor': layers.Anchor,
            'ClipBoxes': layers.ClipBoxes,
            '_smooth_l1': losses.smooth_l1_loss(),
            '_focal': losses.focal_loss(),
        }
        self.validate()

    @abstractmethod
    def create_backbone_model(self, inputs: Input = None, freeze: bool = False) -> Model:
        raise NotImplemented('model property not implemented')

    @abstractmethod
    def download_imagenet(self) -> str:
        """
        Download ImageNet pre-trained model
        :return: pre-trained model path
        """
        raise NotImplementedError('download_imagenet method not implemented.')

    @abstractmethod
    def validate(self):
        raise NotImplementedError('validate method not implemented.')

    @staticmethod
    def freeze(model: Model):
        for layer in model.layers:
            layer.trainable = False
        return model

from abc import ABC, abstractmethod

from keras import Model


class BackboneBase(ABC):

    def __init__(self):
        self.custom_objects = dict()
        self.validate()

    @abstractmethod
    def create_backbone(self):
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

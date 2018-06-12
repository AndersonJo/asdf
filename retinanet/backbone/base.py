from abc import ABC, abstractmethod


class BackboneBase(ABC):

    def __init__(self, backbone):
        self.backbone = backbone
        self.custom_objects = dict()
        self.validate()

    @abstractmethod
    def retinanet(self, *args, **kwargs):
        raise NotImplementedError('retinanet method not implemented.')

    @abstractmethod
    def download_imagenet(self):
        raise NotImplementedError('download_imagenet method not implemented.')

    @abstractmethod
    def validate(self):
        raise NotImplementedError('validate method not implemented.')

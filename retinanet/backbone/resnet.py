import keras_resnet
from keras import Model

from keras.applications.imagenet_utils import get_file
from keras.layers import Input
from keras_resnet.models import ResNet50, ResNet101, ResNet152

from retinanet.backbone.base import BackboneBase


class ResNet50Backbone(BackboneBase):
    def __init__(self):
        super(ResNet50Backbone, self).__init__()
        self.custom_objects.update(keras_resnet.custom_objects)

    def create_backbone_model(self, inputs: Input = None, freeze=False) -> Model:
        if inputs is None:
            inputs = Input(shape=(None, None, 3), name='input')

        model = ResNet50(inputs, include_top=False, freeze_bn=True)
        if freeze:
            model = self.freeze(model)

        return model

    def download_imagenet(self) -> str:
        filename = 'resnet-50.h5'
        resource = 'https://github.com/exemai/sf-retinanet/releases/download/0.0.1/resnet50-keras.h5'
        md5sum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
        model_path = get_file(filename, resource, cache_subdir='models', md5_hash=md5sum)
        return model_path

    def validate(self):
        pass


class ResNet101Backbone(BackboneBase):
    def __init__(self):
        super(ResNet101Backbone, self).__init__()
        self.custom_objects.update(keras_resnet.custom_objects)

    def create_backbone_model(self, inputs: Input = None, freeze=False) -> Model:
        if inputs is None:
            inputs = Input(shape=(None, None, 3), name='input')
        model = ResNet101(inputs, include_top=False, freeze_bn=True)
        if freeze:
            model = self.freeze(model)

        return model

    def download_imagenet(self) -> str:
        filename = 'resnet-101.h5'
        resource = 'https://github.com/exemai/sf-retinanet/releases/download/0.0.1/resnet101-keras.h5'
        md5sum = '05dc86924389e5b401a9ea0348a3213c'
        model_path = get_file(filename, resource, cache_subdir='models', md5_hash=md5sum)
        return model_path

    def validate(self):
        pass


class ResNet152Backbone(BackboneBase):
    def __init__(self):
        super(ResNet152Backbone, self).__init__()
        self.custom_objects.update(keras_resnet.custom_objects)

    def create_backbone_model(self, inputs: Input = None, freeze=False) -> Model:
        if inputs is None:
            inputs = Input(shape=(None, None, 3), name='input')

        model = ResNet152(inputs, include_top=False, freeze_bn=True)
        if freeze:
            model = self.freeze(model)

        return model

    def download_imagenet(self) -> str:
        filename = 'resnet-152.h5'
        resource = 'https://github.com/exemai/sf-retinanet/releases/download/0.0.1/resnet152-keras.h5'
        md5sum = '6ee11ef2b135592f8031058820bb9e71'
        model_path = get_file(filename, resource, cache_subdir='models', md5_hash=md5sum)
        return model_path

    def validate(self):
        pass

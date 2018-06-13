from keras import Model

from retinanet.backbone.base import BackboneBase
from retinanet.backbone.resnet import ResNet50Backbone, ResNet101Backbone, ResNet152Backbone


def load_backbone(backbone: str, freeze=False, weights=None) -> BackboneBase:
    backbone = backbone.lower().strip()
    if backbone == 'resnet50':
        backbone_instance = ResNet50Backbone()
    elif backbone == 'resnet101':
        backbone_instance = ResNet101Backbone()
    elif backbone == 'resnet152':
        backbone_instance = ResNet152Backbone()
    else:
        raise ModelNotFoundError('{0} not found'.format(backbone))

    return backbone_instance


class ModelNotFoundError(Exception):
    pass

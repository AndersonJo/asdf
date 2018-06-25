from PIL.Image import Image
from keras import Input
from keras_resnet.models import ResNet101, ResNet50, ResNet152
from keras.applications.imagenet_utils import get_file

from retinanet.backbone import load_backbone


def test_load_pretrained_weight():
    inputs = Input(shape=(None, None, 3))

    # Resnet 50
    filename = 'resnet-50.h5'
    resource = 'https://github.com/AndersonJo/retinanet-anderson/releases/download/0.0.1/resnet50-keras.h5'
    md5sum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    weights = get_file(filename, resource, cache_subdir='models', md5_hash=md5sum)

    resnet50 = ResNet50(inputs, include_top=False, freeze_bn=True)
    resnet50.load_weights(weights, by_name=True, skip_mismatch=False)

    # Resnet 101
    filename = 'resnet-101.h5'
    resource = 'https://github.com/AndersonJo/retinanet-anderson/releases/download/0.0.1/resnet101-keras.h5'
    md5sum = '05dc86924389e5b401a9ea0348a3213c'
    weights = get_file(filename, resource, cache_subdir='models', md5_hash=md5sum)

    resnet101 = ResNet101(inputs, include_top=False, freeze_bn=True)
    resnet101.load_weights(weights, by_name=True, skip_mismatch=False)

    # Resnet 152
    filename = 'resnet-152.h5'
    resource = 'https://github.com/AndersonJo/retinanet-anderson/releases/download/0.0.1/resnet152-keras.h5'
    md5sum = '6ee11ef2b135592f8031058820bb9e71'
    weights = get_file(filename, resource, cache_subdir='models', md5_hash=md5sum)

    resnet152 = ResNet152(inputs, include_top=False, freeze_bn=True)
    resnet152.load_weights(weights, by_name=True, skip_mismatch=False)

    assert 4 == len(resnet50.outputs)
    assert 4 == len(resnet101.outputs)
    assert 4 == len(resnet152.outputs)


def test_resnet_backbone():
    backbones = ['resnet50', 'resnet101', 'resnet152']

    for name in backbones:
        backbone = load_backbone(name)
        weights = backbone.download_imagenet()
        model = backbone.create_backbone_model()
        model.load_weights(weights, by_name=True, skip_mismatch=False)

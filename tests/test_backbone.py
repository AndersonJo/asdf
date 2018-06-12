from retinanet.backbone.resnet import ResNetBackbone

from keras.applications.

def test_resnet_backbone():
    backbone = ResNetBackbone('resnet50')
    backbone.download_imagenet()
    import ipdb
    ipdb.set_trace()

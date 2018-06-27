import keras.backend as K
from keras.layers import Layer

from retinanet.utils.image import resize_images


class UpSample(Layer):
    """
    Up-sampling refers to any technique that change an image to higher resolution.
    So it could be deconvolution, resizing or etc..

    The UpSampling layer modifies the source image to be the same as the shape of the target image.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)
        return resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

import argparse
import re
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..')))

import keras.backend as K
from keras import Model
from tensorflow.python.saved_model import builder as tf_model_builder, tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

from retinanet.retinanet.model import RetinaNet


def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert training model to inference model')
    parser.add_argument('source', help='The model to convert')
    parser.add_argument('--p2', action='store_true',
                        help='Use P2 of feature pyramid network for detecting tiny objects')
    parser.add_argument('--tensorflow', action='store_true',
                        help='Save as TensorFlow Model for Google Cloud ML')
    return parser.parse_args(args)


def parse_source(source):
    backbone, data_mode, epoch = None, None, None

    regex = re.compile('(?P<backbone>[a-zA-Z\d]+)_?(?P<p2>p2)_(?P<data>\w+)_(?P<epoch>\d+)\.h5')
    match = re.findall(regex, source)

    if match is not None and match:
        backbone, p2, data_mode, epoch = match[0]
        epoch = int(epoch)

    return backbone, data_mode, epoch


def make_destination_path(backbone, data_mode, epochs, tensorflow, use_p2):
    if not os.path.exists('inferences'):
        os.mkdir('inferences')

    p2 = ''
    if use_p2:
        p2 = '_p2'

    dest = 'inferences/{0}{1}_{2}_{3}'.format(backbone, p2, data_mode, epochs)

    if tensorflow:
        dest += '.ckpt'
    else:
        dest += '.h5'
    return dest


# Save Model
def save_as_tensorflow(model: Model, export_path: str):
    builder = tf_model_builder.SavedModelBuilder(export_path)
    signature = predict_signature_def(inputs={'image': model.inputs[0]},
                                      outputs={'boxes': model.outputs[0],
                                               'scores': model.outputs[1],
                                               'labels': model.outputs[2]})
    sess = K.get_session()
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
    )
    builder.save()


def main():
    parser = parse_args(sys.argv[1:])
    backbone, data_mode, epochs = parse_source(parser.source)

    # Set Destination
    dest_path = make_destination_path(backbone, data_mode, epochs, parser.tensorflow, parser.p2)

    # Checks
    if backbone is None or not backbone:
        raise Exception('No backbone ({}) found'.format(backbone))

    # Set Pyramids
    pyramids = ['P3', 'P4', 'P5', 'P6', 'P7']
    if parser.p2:
        pyramids.insert(0, 'P2')

    # Convert training model to inference model
    retinanet = RetinaNet(backbone)
    training_model = retinanet.load_model(parser.source)
    prediction_model = retinanet.create_prediction_model(training_model,
                                                         pyramids=pyramids)

    # Save
    if parser.tensorflow:
        save_as_tensorflow(prediction_model, dest_path)
    else:
        prediction_model.save(dest_path)

    print()
    print(prediction_model.summary(line_length=120))
    print('Pyramids:', pyramids)
    print('Successfully Converted')


if __name__ == '__main__':
    main()

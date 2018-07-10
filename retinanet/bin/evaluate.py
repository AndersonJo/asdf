import argparse
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..')))

from retinanet.utils.eval import Evaluator
from retinanet.retinanet.model import RetinaNet
from retinanet.preprocessing.generator import create_data_generator
from retinanet.preprocessing.pascal import VOC_CLASSES


def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert training model to inference model')
    parser.add_argument('data_mode')
    parser.add_argument('data_path', default='/data/VOCdevkit/')
    parser.add_argument('inference_model', help='The path to Keras inference model')
    parser.add_argument('--convert', action='store_true',
                        help='Convert training model to inference model')
    return parser.parse_args(args)


def parse_model_path(model_path):
    backbone, data_mode, epochs, p2 = None, None, None, None

    regex = re.compile('(?P<backbone>[a-zA-Z\d]+)_?(?P<p2>p2)?_(?P<data_mode>\w+)_(?P<epochs>\d+).h5')
    search = re.findall(regex, model_path)
    if search is not None and search:
        backbone, p2, data_mode, epochs, = search[0]

    use_p2 = False
    if p2:
        use_p2 = True

    return backbone, use_p2, data_mode, epochs


def main():
    parser = parse_args(sys.argv[1:])
    backbone, use_p2, data_mode, epochs = parse_model_path(parser.inference_model)

    if backbone is None or data_mode is None or epochs is None:
        raise Exception('{0} not found'.format(parser.inference_model))

    # Load dataset
    train_generator, test_generator = create_data_generator(parser.data_mode, parser.data_path, batch=1,
                                                            classes=VOC_CLASSES)

    # Set Pyramids and P2
    pyramids = ['P3', 'P4', 'P5', 'P6', 'P7']
    if use_p2:
        pyramids.insert(0, 'P2')

    # Load training model or converted inference model
    # When you load training model, you need to add `--convert` option to convert the training model to inference one
    retinanet = RetinaNet(backbone)
    inference_model = retinanet.load_model(parser.inference_model,
                                           p2=use_p2,
                                           convert=parser.convert)

    # print(inference_model.summary(line_length=120))

    # Evaluate
    evaluator = Evaluator(inference_model, test_generator)
    evaluator()


if __name__ == '__main__':
    main()

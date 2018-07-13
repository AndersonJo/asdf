import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..', '..')))

from retinanet.preprocessing.transform import rescale_image
from retinanet.preprocessing.generator import create_data_generator


def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert training model to inference model')
    parser.add_argument('data_mode')
    parser.add_argument('data_path', default='/data/VOCdevkit/')
    parser.add_argument('--save-file', default='sample.json', help='final JSON file')
    parser.add_argument('--limit', default=5, help='the number of sample images')
    parser.add_argument('--convert', action='store_true',
                        help='Convert training model to inference model')
    return parser.parse_args(args)


def main():
    parser = parse_args(sys.argv[1:])

    # Load dataset
    train_generator, test_generator = create_data_generator(parser.data_mode, parser.data_path)
    generator = test_generator

    # Open file
    f = open(parser.save_file, 'w')

    # Create Samples
    N = min(generator.size(), int(parser.limit))
    for i, idx in enumerate(range(N)):
        annotation_info = generator.get_single_data(idx)
        raw_image = generator.load_image(annotation_info)

        # Normalize the image
        image = generator.preprocess_image(raw_image.copy())

        # Resize Image
        # image, scale_ratio = generator.resize_image(image)
        image, scale = rescale_image(image, 100, 500)

        # Save
        image = image.astype(np.int32)
        # content = json.dumps({'image_bytes': {'b64': base64.b64encode(image).decode('utf-8')}})
        content = json.dumps({'image': image.tolist()})
        f.write(content)
        # if i != (N - 1):
        #     f.write('\n')

        print(annotation_info)

    f.flush()
    f.close()


if __name__ == '__main__':
    main()

from retinanet.preprocessing.pascal import PascalVOCGenerator


def test_generate_anchor():
    voc = PascalVOCGenerator('/data/VOCdevkit', batch=3)
    image_batch, box_batch = voc.get_batch(0)
    voc.process_targets(image_batch, box_batch)
    import ipdb
    ipdb.set_trace()

import re


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

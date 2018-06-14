from retinanet.preprocessing.base import BoundingBoxGenerator


class TestGenerator(BoundingBoxGenerator):

    def __init__(self, data, batch: int = 16, shuffle: bool = True):
        super(TestGenerator, self).__init__(batch=batch, shuffle=shuffle)
        self.data = data

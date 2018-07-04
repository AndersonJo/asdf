from keras.callbacks import Callback

from retinanet.preprocessing.pascal import PascalVOCGenerator


class Evaluate(Callback):

    def __init__(self, generator: PascalVOCGenerator,
                 iou_threshold: float = 0.5,
                 score_threshold: float = 0.05,
                 max_detections: int = 100,
                 save_path: str = None,
                 tensorboard: str = None,
                 verbose: int = 1):

        """
        generator
        :param generator: The generator with dataset to evaluate
        :param iou_threshold: The threshold used to tell positive or negative objects
        :param score_threshold:
        :param max_detections: The maximum number of predictions per image
        :param save_path: The path to save images to visualize predictions
        :param tensorboard: TensorBoard callback instance to log the mAP
        :param verbose: The level of verbosity, 1 is default value
        """

        self.generator = generator
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.save_path = save_path
        self.tensorboard = tensorboard
        self.verbose = verbose

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        average_precisions = self.evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path
        )

        self.mean_ap = sum(average_precisions.values()) / len(average_precisions)

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.mean_ap
            summary_value.tag = "mAP"
            self.tensorboard.writer.add_summary(summary, epoch)

        logs['mAP'] = self.mean_ap

        if self.verbose == 1:
            for label, average_precision in average_precisions.items():
                print(self.generator.label_to_name(label), '{:.4f}'.format(average_precision))
            print('mAP: {:.4f}'.format(self.mean_ap))

    def evaluate(self):
        pass

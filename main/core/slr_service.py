from main.core.algorithm.classifier import Classifier
from main.core.paths import MODEL_DIR
from main.common.utils import log, Timer


class SLRService:
    def __init__(self):
        self.classifier = Classifier(MODEL_DIR)

    def recognize(self, frames):
        log(f"Classifying sequence of {len(frames)} frames.")
        timer = Timer()
        timer.start()
        label, label_index = self.classifier.classify(frames)
        log(f"Classification finished in {timer.get_elapsed_seconds()} seconds.")

        log(f"Result: {label}.")
        return label

from os import path

from joblib import dump


class TrainingResult(object):
    def __init__(self, classifier, dataset):
        self.classifier = classifier
        self.dataset = dataset

    def save(self, output_directory):
        dump(self.classifier, path.join(output_directory, "classifier.joblib"))

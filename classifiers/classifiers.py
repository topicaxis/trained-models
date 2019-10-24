from os import path

from joblib import dump


class TrainingResult(object):
    """Classifier trainig nresult"""

    def __init__(self, classifier, dataset):
        """Create a new TrainingResult object

        :param sklearn.base.BaseEstimator classifier: the trained classifier
        :param classifiers.datasets.Dataset dataset: the dataset that was used
            for training
        """
        self.classifier = classifier
        self.dataset = dataset

    def save(self, output_directory):
        """Save the classifier to the given directory

        :param str output_directory: the output directory
        """
        dump(self.classifier, path.join(output_directory, "classifier.joblib"))

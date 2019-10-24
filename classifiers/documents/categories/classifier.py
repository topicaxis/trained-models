from pkg_resources import resource_filename

import joblib


def create_category_classifier():
    """Create a category classifier using the default model

    :rtype CategoryClassifier
    :return: the category classifier
    """
    binarizer_file = resource_filename(
        "classifiers", "data/document_classifier/binarizer.joblib")
    classifier_file = resource_filename(
        "classifiers", "data/document_classifier/classifier.joblib")

    return CategoryClassifier(
        binarizer=joblib.load(binarizer_file),
        classifier=joblib.load(classifier_file)
    )


class CategoryClassifier(object):
    """Document category classifier object"""
    def __init__(self, binarizer, classifier):
        """Create a new CategoryClassifier object

        :param sklearn.preprocessing.MultiLabelBinarizer binarizer: the
            binarizer to be used
        :param sklearn.base.BaseEstimator classifier: the classifier to be used
        """
        self._binarizer = binarizer
        self._classifier = classifier

    @property
    def classes_(self):
        return self._binarizer.classes_

    def predict(self, data):
        """Predict the document categories

        :param list[str] data: the document contents
        :rtype: list[str]
        :return: the predicted document categories
        """
        result = self._classifier.predict(data)
        return self._binarizer.inverse_transform(result)

    def predict_proba(self, data):
        """Predict the category probabilities for the given documents

        :param list[str] data: the document contents
        :rtype: list[list[float]]
        :return: the probability for each category
        """
        return self._classifier.predict_proba(data)

from pkg_resources import resource_filename

import joblib


def create_category_classifier():
    binarizer_file = resource_filename(
        "classifiers", "data/document_classifier/binarizer.joblib")
    classifier_file = resource_filename(
        "classifiers", "data/document_classifier/classifier.joblib")

    return CategoryClassifier(
        binarizer=joblib.load(binarizer_file),
        classifier=joblib.load(classifier_file)
    )


class CategoryClassifier(object):
    def __init__(self, binarizer, classifier):
        self._binarizer = binarizer
        self._classifier = classifier

    @property
    def classes_(self):
        return self._binarizer.classes_

    def predict(self, data):
        result = self._classifier.predict(data)
        return self._binarizer.inverse_transform(result)

    def predict_proba(self, data):
        return self._classifier.predict_proba(data)

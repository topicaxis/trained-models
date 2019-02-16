from pkg_resources import resource_filename

from sklearn.externals import joblib


class CategoryClassifier(object):
    def __init__(self):
        binarizer_file = resource_filename(
            "classifiers", "data/label_binarizer/label_binarizer.pickle")
        classifier_file = resource_filename(
            "classifiers", "data/trained_classifier/classifier.pickle")

        self._binarizer = joblib.load(binarizer_file)
        self._classifier = joblib.load(classifier_file)

    @property
    def classes_(self):
        return self._binarizer.classes_

    def predict(self, data):
        result = self._classifier.predict(data)
        return self._binarizer.inverse_transform(result)

    def predict_proba(self, data):
        return self._classifier.predict_proba(data)

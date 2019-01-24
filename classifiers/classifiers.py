from pkg_resources import resource_filename
import re

import numpy as np
from sklearn.externals import joblib
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer


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


class KeywordSelectionClassifier(Pipeline):
    """Keyword selection classifier

    The result from the predict method of the classifier has the following
    meaning

    0 = the keyword is noise
    1 = the keyword is valid and can be used
    """

    def __init__(self):
        self._alnum_re = re.compile(r"\w+")
        self._num_re = re.compile(r"\d+")
        self._other_re = re.compile(r"[\W]+")
        self._whitespace_re = re.compile(r"[\s]+")

        keyword_to_metrics = FunctionTransformer(self._extract_keyword_metrics)

        vectorizer_file = resource_filename(
            "classifiers",
            "data/keyword_selection_classifier/countvectorizer.joblib"
        )
        vectorizer = joblib.load(vectorizer_file)

        feature_extractor = FeatureUnion([
            ("keyword_metrics", keyword_to_metrics),
            ("vectorizer", vectorizer)
        ])

        feature_selector_file = resource_filename(
            "classifiers",
            "data/keyword_selection_classifier/feature_selector.joblib"
        )
        feature_selector = joblib.load(feature_selector_file)

        classifier_file = resource_filename(
            "classifiers",
            "data/keyword_selection_classifier/classifier.joblib"
        )
        classifier = joblib.load(classifier_file)

        super(KeywordSelectionClassifier, self).__init__([
            ("feature_extractor", feature_extractor),
            ("feature_selector", feature_selector),
            ("classifier", classifier)
        ])

    def _extract_keyword_metrics(self, X):
        features = []

        for keyword in X[0]:
            features.append([
                len(keyword),
                len(keyword.split(" ")),
                sum([len(item) for item in self._alnum_re.findall(keyword)]),
                sum([len(item) for item in self._num_re.findall(keyword)]),
                sum([len(item) for item in self._other_re.findall(keyword)]),
                sum([len(item)
                     for item in self._whitespace_re.findall(keyword)]),
                max([len(x) for x in keyword.split(" ")]),
                min([len(x) for x in keyword.split(" ")])
            ])

        return np.array(features)

from pkg_resources import resource_filename

import joblib


def create_keyword_selection_classifier():
    classifier_file = resource_filename(
        "classifiers",
        "data/keyword_selection_classifier/classifier.joblib"
    )

    return KeywordSelectionClassifier(joblib.load(classifier_file))


class KeywordSelectionClassifier(object):
    """Keyword selection classifier

    The result from the predict method of the classifier has the following
    meaning

    0 = the keyword is noise
    1 = the keyword is valid and can be used
    """

    def __init__(self, classifier):
        self._classifier = classifier

    def predict(self, keywords):
        return self._classifier.predict(keywords)

    def predict_proba(self, keywords):
        return self._classifier.predict_proba(keywords)

    def select_keywords(self, keywords):
        predictions = self.predict(keywords)

        return [
            keyword
            for keyword, score in zip(keywords, predictions)
            if score == 1
        ]

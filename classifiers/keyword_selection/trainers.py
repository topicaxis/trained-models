from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from classifiers.classifiers import TrainingResult
from classifiers.keyword_selection.features.extraction import (
    KeywordSelectionFeatureExtractor
)


class KeywordSelectionClassifierTrainer(object):
    """Keyword selection classifier trainer"""

    def __init__(self, dataset):
        """Create a new KeywordSelectionClassifierTrainer object

        :param Dataset dataset: the dataset that will be used for training
        """
        self._dataset = dataset

    def _create_pipeline(self, n_estimators, max_depth, max_features):
        return Pipeline([
            ("vectorizer", KeywordSelectionFeatureExtractor()),
            ("classifier", RandomForestClassifier(n_estimators=n_estimators,
                                                  max_depth=max_depth,
                                                  max_features=max_features,
                                                  random_state=42))
        ])

    def train(self, n_estimators, max_depth, max_features):
        """Train a classifier

        :param int n_estimators: the number of entimators to train
        :param int max_depth: the random forest classifier max depth
        :param int max_features: the random forest classifier max number of
            features
        :rtype: TrainingResult
        :return: the training result
        """
        pipeline = self._create_pipeline(n_estimators, max_depth, max_features)

        pipeline.fit(self._dataset.data, self._dataset.targets)

        return TrainingResult(pipeline, self._dataset)

from os import path

from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from classifiers.datasets import Dataset


class CategoryClassifierTrainingResult(object):
    """Document category classifier training result object"""

    def __init__(self, binarizer, classifier, dataset):
        """Create a new CategoryClassifierTrainingResult object

        :param MultiLabelBinarizer binarizer: the
            binarizer to be used
        :param sklearn.base.BaseEstimator classifier: the classifier to be used
        :param Dataset dataset: the dataset that will be used for training
        """
        self.binarizer = binarizer
        self.classifier = classifier
        self.dataset = dataset

    def save(self, output_directory):
        """Save the classifier component to the given directory

        :param str output_directory: the output directory
        """
        dump(self.classifier, path.join(output_directory, "classifier.joblib"))
        dump(self.binarizer, path.join(output_directory, "binarizer.joblib"))


class CategoryClassifierTrainer(object):
    """Document category classifier trainer object"""

    def __init__(self, dataset):
        """Create a new CategoryClassifierTrainer object

        :param Dataset dataset: the dataset that will be used for training
        """
        self._dataset = dataset

    @staticmethod
    def _create_pipeline(max_df, min_df, feature_count, penalty, c):
        return Pipeline([
            ("vectorizer", CountVectorizer(
                ngram_range=(1, 1), max_df=max_df, min_df=min_df)),
            ("feature_selection", SelectKBest(chi2, k=feature_count)),
            ("classifier", OneVsRestClassifier(
                LogisticRegression(penalty=penalty, C=c, random_state=42,
                                   max_iter=1000, solver="liblinear")))
        ])

    def _create_binarizer(self):
        binarizer = MultiLabelBinarizer(
            classes=list(set(self._dataset.targets)))
        binarizer.fit(self._dataset.targets)

        return binarizer

    def _prepare_dataset(self, binarizer):
        binarized_categories = binarizer.transform(
            [[item] for item in self._dataset.targets])

        return Dataset(data=self._dataset.data, targets=binarized_categories)

    def train(self, max_df, min_df, feature_count, penalty, c):
        """Train a classifier

        :param float max_df: the maximum document frequency in 0-1 range
        :param int min_df: the minimum document frequency
        :param int feature_count: the number of features to select
        :param str penalty: the training penalty to use. This can be l1 or l2
        :param float c: the C value for the logistic regression classifier
        :rtype: CategoryClassifierTrainingResult
        :return: the training result
        """
        binarizer = self._create_binarizer()
        prepared_dataset = self._prepare_dataset(binarizer)

        pipeline = self._create_pipeline(
            max_df=max_df,
            min_df=min_df,
            feature_count=feature_count,
            penalty=penalty,
            c=c
        )
        pipeline.fit(prepared_dataset.data, prepared_dataset.targets)

        return CategoryClassifierTrainingResult(
            binarizer=binarizer,
            classifier=pipeline,
            dataset=self._dataset
        )

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
    def __init__(self, binarizer, pipeline, dataset):
        self.binarizer = binarizer
        self.pipeline = pipeline
        self.dataset = dataset

    def save(self, output_directory):
        dump(self.pipeline, path.join(output_directory, "classifier.joblib"))
        dump(self.binarizer, path.join(output_directory, "binarizer.joblib"))


class CategoryClassifierTrainer(object):
    def __init__(self, dataset):
        self._dataset = dataset

    @staticmethod
    def _create_pipeline(max_df, min_df, feature_count, penalty, c):
        return Pipeline([
            ("vectorizer", CountVectorizer(ngram_range=(1, 1), max_df=max_df, min_df=min_df)),
            ("feature_selection", SelectKBest(chi2, k=feature_count)),
            ("classifier", OneVsRestClassifier(LogisticRegression(penalty=penalty, C=c, random_state=42)))
        ])

    def _create_binarizer(self):
        binarizer = MultiLabelBinarizer(classes=list(set(self._dataset.targets)))
        binarizer.fit(self._dataset.targets)

        return binarizer

    def _prepare_dataset(self, binarizer):
        binarized_categories = binarizer.transform([[item] for item in self._dataset.targets])

        return Dataset(data=self._dataset.data, targets=binarized_categories)

    def train(self, max_df, min_df, feature_count, penalty, c):
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
            pipeline=pipeline,
            dataset=self._dataset
        )

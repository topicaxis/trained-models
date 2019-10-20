from collections import namedtuple
import json

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

from classifiers.datasets import TrainTestDataset, Dataset


CategoryClassifierTrainingResult = namedtuple(
    "CategoryClassifierTrainingResult",
    ["binarizer", "pipeline", "train_test_dataset"]
)


class CategoryClassifierTrainer(object):
    def __init__(self, dataset_file, test_size=0.3, min_text_length=200, selected_categories=None):
        self._dataset_file = dataset_file
        self._test_size = test_size
        self._min_text_length = min_text_length
        self._selected_categories = selected_categories

        self._load_dataset()

    def _can_use_data(self, data):
        if self._selected_categories and data["category"] not in self._selected_categories:
            return False

        return data["text"] and len(data["text"]) >= self._min_text_length

    def _load_dataset(self):
        self._documents = []
        self._categories = []

        with open(self._dataset_file) as f:
            for line in f:
                data = json.loads(line)
                if self._can_use_data(data):
                    self._documents.append(data["text"])
                    self._categories.append(data["category"])

    def _create_pipeline(self, max_df, min_df, feature_count, penalty, c):
        return Pipeline([
            ("vectorizer", CountVectorizer(ngram_range=(1, 1), max_df=max_df, min_df=min_df)),
            ("feature_selection", SelectKBest(chi2, k=feature_count)),
            ("classifier", OneVsRestClassifier(LogisticRegression(penalty=penalty, C=c, random_state=42)))
        ])

    def _create_binarizer(self):
        binarizer = MultiLabelBinarizer(classes=list(set(self._categories)))
        binarizer.fit(self._categories)

        return binarizer

    def _create_train_test_dataset(self, binarizer):
        train_documents, test_documents, train_categories, test_categories = train_test_split(self._documents, self._categories, test_size=self._test_size, random_state=42, stratify=self._categories)

        train_categories = binarizer.transform([[item] for item in train_categories])
        test_categories = binarizer.transform([[item] for item in test_categories])

        return TrainTestDataset(
            train=Dataset(data=train_documents, targets=train_categories),
            test=Dataset(data=test_documents, targets=test_categories)
        )

    def train(self, max_df, min_df, feature_count, penalty, c):
        binarizer = self._create_binarizer()
        train_test_dataset = self._create_train_test_dataset(binarizer)

        pipeline = self._create_pipeline(
            max_df=max_df,
            min_df=min_df,
            feature_count=feature_count,
            penalty=penalty,
            c=c
        )
        pipeline.fit(train_test_dataset.train.data, train_test_dataset.train.targets)

        return CategoryClassifierTrainingResult(
            binarizer=binarizer,
            pipeline=pipeline,
            train_test_dataset=train_test_dataset
        )

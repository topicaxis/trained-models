import csv

from sklearn.model_selection import train_test_split

from classifiers.datasets import TrainTestDataset, Dataset


class KeywordSelectionClassifierDatasetLoader(object):
    def __init__(self, dataset_file):
        self._dataset_file = dataset_file

    def _load_dataset(self):
        keywords = []
        is_valid = []

        with open(self._dataset_file) as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            next(reader)  # skip the header
            for row in reader:
                keywords.append(row[0])
                is_valid.append(int(row[1]))

        return keywords, is_valid

    def create_train_test_dataset(self, test_size=0.3):
        keywords, is_valid = self._load_dataset()

        train_keywords, test_keywords, train_is_valid, test_is_valid = train_test_split(keywords, is_valid, test_size=test_size, random_state=42, stratify=is_valid)

        return TrainTestDataset(
            train=Dataset(data=train_keywords, targets=train_is_valid),
            test=Dataset(data=test_keywords, targets=test_is_valid)
        )

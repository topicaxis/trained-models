import json

from sklearn.model_selection import train_test_split

from classifiers.datasets import TrainTestDataset, Dataset


class CategoryClassifierDatasetLoader(object):
    """Document category classifier dataset loader"""

    def __init__(self, dataset_file):
        """Create a new CategoryClassifierDatasetLoader object

        The dataset file must contain a json object in each line. The item must
        contain a 'text' and a 'category' attribute.

        :param str dataset_file: the path to the dataset file
        """
        self._dataset_file = dataset_file

    @staticmethod
    def _can_use_data(data, selected_categories, min_text_length):
        if selected_categories and data["category"] not in selected_categories:
            return False

        return data["text"] and len(data["text"]) >= min_text_length

    def _load_dataset(self, selected_categories, min_text_length):
        documents = []
        categories = []

        with open(self._dataset_file) as f:
            for line in f:
                data = json.loads(line)
                if self._can_use_data(data, selected_categories,
                                      min_text_length):
                    documents.append(data["text"])
                    categories.append(data["category"])

        return documents, categories

    def create_train_test_dataset(self, test_size=0.3, min_text_length=200,
                                  selected_categories=None):
        """Create a training/testing dataset

        :param float test_size: the test size percentage in the 0-1 value range
        :param int min_text_length: the minimun text length that a document
            must be in order to be used
        :param list[str]|None selected_categories: which categories to use in
            the dataset
        :rtype: TrainTestDataset
        :return: the training/testing dataset
        """
        documents, categories = self._load_dataset(
            selected_categories, min_text_length)

        train_documents, test_documents, train_categories, test_categories = \
            train_test_split(documents, categories, test_size=test_size,
                             random_state=42, stratify=categories)

        return TrainTestDataset(
            train=Dataset(data=train_documents, targets=train_categories),
            test=Dataset(data=test_documents, targets=test_categories)
        )

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from classifiers.keyword_selection import feature_functions


class KeywordSelectionFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._alnum_re = re.compile(r"\w+")
        self._num_re = re.compile(r"\d+")
        self._other_re = re.compile(r"[\W]+")
        self._whitespace_re = re.compile(r"[\s]+")

        self._feature_functions = {
            "keyword_length": feature_functions.keyword_length,
            "word_count": feature_functions.word_count,
            "alphanumeric_count": feature_functions.alphanumeric_count,
            "numeric_count": feature_functions.numeric_count,
            "other_character_count": feature_functions.other_character_count,
            "whitespace_count": feature_functions.whitespace_count,
            "max_word_length": feature_functions.max_word_length,
            "min_word_length": feature_functions.min_word_length
        }

        self._keyword_length_ratio_feature_functions = {
            "alphanumeric_ratio": "alphanumeric_count",
            "numeric_ratio": "numeric_count",
            "other_character_ratio": "other_character_count",
            "whitespace_ratio": "whitespace_count",
        }

    @property
    def feature_names(self):
        return sorted(set(
            list(self._feature_functions.keys()) +
            list(self._keyword_length_ratio_feature_functions.keys()) +
            ["max_min_word_length_ratio", "non_alphanumeric_ratio"]
        ))

    def extract_features(self, keywords):
        features = [
            {
                feature_name: feature_function(keyword)
                for feature_name, feature_function in
                self._feature_functions.items()
            }
            for keyword in keywords
        ]

        keyword_length_ratio_features = [
            {
                feature_name: feature_functions.calculate_keyword_length_ratio(
                    feature_instance, source_feature_name)
                for feature_name, source_feature_name in
                self._keyword_length_ratio_feature_functions.items()
            }
            for feature_instance in features
        ]

        for feature, keyword_length_ratio_feature in zip(
                features, keyword_length_ratio_features):
            feature.update(keyword_length_ratio_feature)

        for feature in features:
            feature["max_min_word_length_ratio"] = round((feature["min_word_length"] / float(feature["max_word_length"])) * 100.0)  # noqa
            feature["non_alphanumeric_ratio"] = round(((feature["other_character_count"] + feature["whitespace_count"]) / float(feature["keyword_length"])) * 100.0)  # noqa

        return features

    def extract_features_array(self, keywords):
        features = self.extract_features(keywords)

        feature_names = self.feature_names

        return [
            [
                feature_instance[feature_name]
                for feature_name in feature_names
            ]
            for feature_instance in features
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, keywords):
        features = self.extract_features_array(keywords)

        return np.array(features)

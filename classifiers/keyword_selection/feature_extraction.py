import re

import numpy as np


class KeywordSelectionFeatureExtractor(object):
    def __init__(self):
        self._alnum_re = re.compile(r"\w+")
        self._num_re = re.compile(r"\d+")
        self._other_re = re.compile(r"[\W]+")
        self._whitespace_re = re.compile(r"[\s]+")

        self._feature_functions = {
            "keyword_length": lambda keyword: len(keyword),
            "word_count": lambda keyword: len(keyword.split(" ")),
            "alphanumeric_count": lambda keyword: sum([len(item) for item in self._alnum_re.findall(keyword)]),
            "numeric_count": lambda keyword: sum([len(item) for item in self._num_re.findall(keyword)]),
            "other_character_count": lambda keyword: sum([len(item) for item in self._other_re.findall(keyword)]),
            "whitespace_count": lambda keyword: sum([len(item) for item in self._whitespace_re.findall(keyword)]),
            "max_word_length": lambda keyword: max([len(x) for x in keyword.split(" ")]),
            "min_word_length": lambda keyword: min([len(x) for x in keyword.split(" ")])
        }

        self._keyword_length_ratio_func = lambda features, feature_name: round((features[feature_name] / float(features["keyword_length"])) * 100.0)
        self._keyword_length_ratio_feature_functions = {
            "alphanumeric_ratio": "alphanumeric_count",
            "numeric_ratio": "numeric_count",
            "other_character_ratio": "other_character_count",
            "whitespace_ratio": "whitespace_count",
        }

    @property
    def feature_names(self):
        return sorted(set(self._feature_functions.keys() + self._keyword_length_ratio_feature_functions.keys() + ["max_min_word_length_ratio", "non_alphanumeric_ratio"]))

    def extract_features(self, keywords):
        features = [
            {
                feature_name: feature_function(keyword)
                for feature_name, feature_function in self._feature_functions.items()
            }
            for keyword in keywords
        ]

        keyword_length_ratio_features = [
            {
                feature_name: self._keyword_length_ratio_func(feature_instance, source_feature_name)
                for feature_name, source_feature_name in self._keyword_length_ratio_feature_functions.items()
            }
            for feature_instance in features
        ]

        for feature, keyword_length_ratio_feature in zip(features, keyword_length_ratio_features):
            feature.update(keyword_length_ratio_feature)

        for feature in features:
            feature["max_min_word_length_ratio"] = round((feature["min_word_length"] / float(feature["max_word_length"])) * 100.0)
            feature["non_alphanumeric_ratio"] = round(((feature["other_character_count"] + feature["whitespace_count"]) / float(feature["keyword_length"])) * 100.0)

        return features

    def extract_features_array(self, keywords):
        features = self.extract_features(keywords)

        feature_names = self.feature_names

        return np.array([
            [
                feature_instance[feature_name]
                for feature_name in feature_names
            ]
            for feature_instance in features
        ])

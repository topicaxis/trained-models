import re


__alnum_re = re.compile(r"\w+")
__num_re = re.compile(r"\d+")
__other_re = re.compile(r"[\W]+")
__whitespace_re = re.compile(r"[\s]+")


def keyword_length(keyword):
    return len(keyword)


def word_count(keyword):
    return len(keyword.split(" "))


def alphanumeric_count(keyword):
    return sum([len(item) for item in __alnum_re.findall(keyword)])


def numeric_count(keyword):
    return sum([len(item) for item in __num_re.findall(keyword)])


def other_character_count(keyword):
    return sum([len(item) for item in __other_re.findall(keyword)])


def whitespace_count(keyword):
    return sum([len(item) for item in __whitespace_re.findall(keyword)])


def max_word_length(keyword):
    return max([len(x) for x in keyword.split(" ")])


def min_word_length(keyword):
    return min([len(x) for x in keyword.split(" ")])


def calculate_keyword_length_ratio(features, feature_name):
    return round((features[feature_name] / float(features["keyword_length"])) * 100.0)

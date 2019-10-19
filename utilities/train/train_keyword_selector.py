from argparse import ArgumentParser
import csv

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

from classifiers.keyword_selection.feature_extraction import KeywordSelectionFeatureExtractor


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--classifier", default="classifier.joblib")
    parser.add_argument("--n-estimators", default=10, type=int)
    parser.add_argument("--max-depth", default=4, type=int)
    parser.add_argument("--max-features", default=4, type=int)
    parser.add_argument("--test-size", default=0.3, type=float)

    return parser.parse_args()


def load_dataset(data_file):
    keywords = []
    is_valid = []

    with open(data_file) as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        next(reader)
        for row in reader:
            keywords.append(row[0])
            is_valid.append(int(row[1]))

    return keywords, is_valid


def prepare_dataset(keywords, is_valid, test_size):
    return train_test_split(keywords, is_valid, test_size=test_size, random_state=42, stratify=is_valid)


def train_classifier(keywords, is_valid, n_estimators, max_depth, max_features):
    pipeline = Pipeline([
        ("vectorizer", KeywordSelectionFeatureExtractor()),
        ("classifier", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=42))
    ])
    pipeline.fit(keywords, is_valid)

    return pipeline


def evaluate_classifier(classifier, keywords, is_valid):
    predicted = classifier.predict(keywords)

    print(classification_report(is_valid, predicted))


def main():
    args = get_arguments()

    keywords, is_valid = load_dataset(args.dataset)

    train_keywords, test_keywords, train_is_valid, test_is_valid = prepare_dataset(keywords, is_valid, args.test_size)

    classifier = train_classifier(
        keywords=train_keywords,
        is_valid=train_is_valid,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=args.max_features
    )

    evaluate_classifier(classifier, test_keywords, test_is_valid)

    dump(classifier, args.classifier)


if __name__ == "__main__":
    main()

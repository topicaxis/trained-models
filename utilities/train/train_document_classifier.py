from argparse import ArgumentParser
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


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--classifier", default="classifier.joblib")
    parser.add_argument("--binarizer", default="binarizer.joblib")
    parser.add_argument("--min-text-length", default=200, type=int)
    parser.add_argument("--test-size", default=0.3, type=float)
    parser.add_argument("--feature-count", default=5000, type=int)
    parser.add_argument("--max-df", default=0.6, type=float)
    parser.add_argument("--min-df", default=2, type=int)
    parser.add_argument("-c", default=1.0, type=float)
    parser.add_argument("--penalty", default="l2")

    return parser.parse_args()


def load_dataset(data_file, min_text_length):
    documents = []
    categories = []

    with open(data_file) as f:
        for line in f:
            data = json.loads(line)
            if data["text"] and len(data["text"]) >= min_text_length and data["category"]:
                documents.append(data["text"])
                categories.append(data["category"])

    return documents, categories


def create_binarizer(categories):
    binarizer = MultiLabelBinarizer(classes=list(set(categories)))
    binarizer.fit(categories)

    return binarizer


def prepare_dataset(binarizer, documents, categories, test_size):
    train_documents, test_documents, train_categories, test_categories = train_test_split(documents, categories, test_size=test_size, random_state=42, stratify=categories)

    train_categories = binarizer.transform([[item] for item in train_categories])
    test_categories = binarizer.transform([[item] for item in test_categories])

    return train_documents, test_documents, train_categories, test_categories


def train_classifier(documents, categories, max_df, min_df, feature_count, penalty, c):
    pipeline = Pipeline([
        ("vectorizer",CountVectorizer(ngram_range=(1, 1), max_df=max_df, min_df=min_df)),
        ("feature_selection", SelectKBest(chi2, k=feature_count)),
        ("classifier", OneVsRestClassifier(LogisticRegression(penalty=penalty, C=c, random_state=42)))
    ])
    pipeline.fit(documents, categories)

    return pipeline


def evaluate_classifier(classifier, documents, categories):
    predicted = classifier.predict(documents)

    print(classification_report(categories, predicted))


def main():
    args = get_arguments()

    documents, categories = load_dataset(args.dataset, args.min_text_length)

    binarizer = create_binarizer(categories)

    train_documents, test_documents, train_categories, test_categories = prepare_dataset(binarizer, documents, categories, args.test_size)

    classifier = train_classifier(
        documents=train_documents,
        categories=train_categories,
        max_df=args.max_df,
        min_df=args.min_df,
        feature_count=args.feature_count,
        penalty=args.penalty,
        c=args.c
    )

    evaluate_classifier(classifier, test_documents, test_categories)

    dump(classifier, args.classifier)
    dump(binarizer, args.binarizer)


if __name__ == "__main__":
    main()

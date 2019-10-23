from argparse import ArgumentParser

from classifiers.documents.categories.classifier import create_category_classifier


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("document")

    return parser.parse_args()


def main():
    args = get_arguments()

    classifier = create_category_classifier()

    with open(args.document) as f:
        contents = f.read()

    result = classifier.predict_proba([contents])

    for category, probability in zip(classifier.classes_, result[0]):
        print(category, probability)


if __name__ == "__main__":
    main()

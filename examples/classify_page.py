from argparse import ArgumentParser

import requests
from bs4 import BeautifulSoup

from classifiers.classifiers import CategoryClassifier


def process_web_page(content):
    soup = BeautifulSoup(content, 'html.parser')
    [s.extract() for s in soup.findAll("script")]
    [s.extract() for s in soup.findAll("style")]
    return soup.get_text()


def get_arguments():
    parser = ArgumentParser()

    parser.add_argument("classifier")
    parser.add_argument("binarizer")
    parser.add_argument("url")

    return parser.parse_args()


def main():
    args = get_arguments()

    classifier = CategoryClassifier()

    response = requests.get(args.url)

    if response.status_code != 200:
        print("Failed to download content")
        exit()

    contents = process_web_page(response.text).lower()

    result = classifier.predict_proba([contents])

    for category, probability in zip(classifier.classes_, result[0]):
        print(category, probability)


if __name__ == "__main__":
    main()

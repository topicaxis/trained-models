from classifiers.keyword_selection.classifier import KeywordSelectionClassifier


def main():
    classifier = KeywordSelectionClassifier()

    keywords = [
        "$# ^%$ sdfs #",
        "machine learning python package"
    ]

    predictions = classifier.select_keywords(keywords)

    print(predictions)


if __name__ == "__main__":
    main()

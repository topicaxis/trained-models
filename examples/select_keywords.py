from classifiers.keyword_selection.classifiers import create_keyword_selection_classifier


def main():
    classifier = create_keyword_selection_classifier()

    keywords = [
        "$# ^%$ sdfs #",
        "machine learning python package"
    ]

    predictions = classifier.select_keywords(keywords)

    print(predictions)


if __name__ == "__main__":
    main()

from argparse import ArgumentParser

from sklearn.metrics import classification_report

from classifiers.keyword_selection.datasets.loaders import KeywordSelectionClassifierDatasetLoader
from classifiers.keyword_selection.trainers import KeywordSelectionClassifierTrainer


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output", default="output")
    parser.add_argument("--n-estimators", default=10, type=int)
    parser.add_argument("--max-depth", default=4, type=int)
    parser.add_argument("--max-features", default=4, type=int)
    parser.add_argument("--test-size", default=0.3, type=float)

    return parser.parse_args()


def evaluate_classifier(classifier, keywords, is_valid):
    predicted = classifier.predict(keywords)

    print(classification_report(is_valid, predicted))


def main():
    args = get_arguments()

    dataset_loader = KeywordSelectionClassifierDatasetLoader(args.dataset)
    dataset = dataset_loader.create_train_test_dataset(args.test_size)

    trainer = KeywordSelectionClassifierTrainer(dataset.train)

    training_result = trainer.train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=args.max_features
    )

    evaluate_classifier(training_result.classifier, dataset.test.data, dataset.test.targets)
    training_result.save(args.output)


if __name__ == "__main__":
    main()

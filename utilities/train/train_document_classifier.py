from argparse import ArgumentParser

from sklearn.metrics import classification_report

from classifiers.documents.categories.trainer import CategoryClassifierTrainer
from classifiers.documents.categories.datasets.loaders import CategoryClassifierDatasetLoader


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output", default="output")
    parser.add_argument("--min-text-length", default=200, type=int)
    parser.add_argument("--test-size", default=0.3, type=float)
    parser.add_argument("--feature-count", default=5000, type=int)
    parser.add_argument("--max-df", default=0.6, type=float)
    parser.add_argument("--min-df", default=2, type=int)
    parser.add_argument("-c", default=1.0, type=float)
    parser.add_argument("--penalty", default="l2")

    return parser.parse_args()


def evaluate_classifier(classifier, documents, categories):
    predicted = classifier.predict(documents)

    print(classification_report(categories, predicted))


def main():
    args = get_arguments()

    dataset_loader = CategoryClassifierDatasetLoader(args.dataset)
    train_test_dataset = dataset_loader.create_train_test_dataset(
        test_size=args.test_size,
        min_text_length=args.min_text_length,
        selected_categories=["cat__programming", "cat__science", "cat__business", "cat__politics", "cat__technology"]
    )

    trainer = CategoryClassifierTrainer(train_test_dataset.train)
    training_result = trainer.train(
        max_df=args.max_df,
        min_df=args.min_df,
        feature_count=args.feature_count,
        penalty=args.penalty,
        c=args.c
    )

    # convert the test targets into an indicator matrix
    binarized_test_targets = training_result.binarizer.transform([[item] for item in train_test_dataset.test.targets])

    evaluate_classifier(training_result.pipeline, train_test_dataset.test.data, binarized_test_targets)
    training_result.save(args.output)


if __name__ == "__main__":
    main()

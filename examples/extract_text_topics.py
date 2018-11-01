from argparse import ArgumentParser

from classifiers.topics import get_text_topics_extractor


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("text_file")

    return parser.parse_args()


def main():
    args = get_arguments()

    text_topics_extractor = get_text_topics_extractor()

    with open(args.text_file) as f:
        topics = text_topics_extractor.predict_topics([f.read()])
        print(topics[0])


if __name__ == "__main__":
    main()

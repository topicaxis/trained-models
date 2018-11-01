from pkg_resources import resource_filename

from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


def get_text_topics_extractor():
    vectorizer_file = resource_filename(
        "classifiers", "data/text_topics/vectorizer.joblib")
    vectorizer = joblib.load(vectorizer_file)

    lda_file = resource_filename(
        "classifiers", "data/text_topics/lda.joblib")
    lda = joblib.load(lda_file)

    pipeline = Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("lda", lda)
        ]
    )

    return TextTopics(pipeline)


class TextTopics(object):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @property
    def topic_count(self):
        num_topics, _ = self.pipeline.named_steps["lda"].components_.shape

        return num_topics

    def __top_keyword_scores(self, topic, num_keywords):
        keyword_scores = sorted(
            enumerate(self.pipeline.named_steps["lda"].components_[topic, :]),
            key=lambda item: item[1],
            reverse=True
        )

        return keyword_scores[:num_keywords]

    def predict(self, data):
        return self.pipeline.transform(data)

    def topic_keywords(self, topic, num_keywords=20):
        top_keyword_scores = self.__top_keyword_scores(topic, num_keywords)
        top_keywords = \
            [keyword_index for keyword_index, _ in top_keyword_scores]
        feature_names = \
            self.pipeline.named_steps["vectorizer"].get_feature_names()

        return [feature_names[keyword_index] for keyword_index in top_keywords]

    def topic_keyword_scores(self, topic, num_keywords=20):
        top_keyword_scores = self.__top_keyword_scores(topic, num_keywords)

        feature_names = \
            self.pipeline.named_steps["vectorizer"].get_feature_names()

        return {
            feature_names[keyword_index]: score
            for keyword_index, score in top_keyword_scores
        }

    def topics(self, num_keywords=20):
        return [
            self.topic_keyword_scores(topic, num_keywords)
            for topic in range(self.topic_count)
        ]

    def predict_topics(self, data, num_topics=5, num_keywords=20):
        topic_districutions = self.predict(data)

        results = []

        for prediction in topic_districutions:
            topic_scores = enumerate(prediction)
            popular_topics = sorted(
                topic_scores,
                key=lambda item: item[1],
                reverse=True
            )[:num_topics]

            results.append([
                {
                    "id": topic_id,
                    "score": score,
                    "keywords": self.topic_keyword_scores(
                        topic_id, num_keywords)
                }
                for topic_id, score in popular_topics
            ])

        return results

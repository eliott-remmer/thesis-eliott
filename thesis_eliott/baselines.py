import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline


def dummy(labels, strategy="most_frequent", random_state=0):
    classifier = DummyClassifier(strategy=strategy, random_state=random_state,)
    classifier.fit(np.ones((len(labels), 1)), labels)
    return classifier


def bow_linear(inputs, labels):
    classifier = LogisticRegression(solver="lbfgs", max_iter=2000,)
    char_vectorizer = TfidfVectorizer(
        min_df=0.001,
        max_df=0.8,
        ngram_range=[1, 4],
        use_idf=True,
        norm="l2",
        analyzer="char",
        smooth_idf=True,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    word_vectorizer = TfidfVectorizer(
        min_df=0.001,
        max_df=0.8,
        ngram_range=[1, 2],
        use_idf=True,
        norm="l2",
        analyzer="word",
        smooth_idf=True,
        strip_accents="unicode",
        sublinear_tf=True,
    )
    features = FeatureUnion([("chars", char_vectorizer), ("words", word_vectorizer)])
    classifier_pipeline = make_pipeline(features, classifier)
    classifier_pipeline.fit(inputs, labels)
    return classifier_pipeline

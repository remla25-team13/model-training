"""Train model(s)"""

import joblib
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from dvclive import Live


def train(live=None):
    """Run training step"""
    if live:
        live.log_param("classifiers", "GaussianNB, BernouliNB")

    x = joblib.load('output/splits/X_train.jbl')
    y = joblib.load('output/splits/y_train.jbl')

    gauss_classifier = GaussianNB()
    multi_classifier = MultinomialNB()
    gauss_classifier.fit(x, y)
    multi_classifier.fit(x, y)

    joblib.dump(gauss_classifier, 'output/model-gauss.jbl')
    joblib.dump(multi_classifier, 'output/model-multi.jbl')

    if live:
        live.log_artifact("output/model-gauss.jbl", type="model")
        live.log_artifact("output/model-multi.jbl", type="model")


if __name__ == "__main__":
    with Live() as live:
        train(live)

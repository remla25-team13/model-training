"""Train model(s)"""

import joblib
from dvclive import Live
from sklearn.naive_bayes import GaussianNB, BernoulliNB


def train():
    """Run training step"""
    with Live() as live:
        live.log_param("classifiers", "GaussianNB, BernouliNB")

        x = joblib.load('output/splits/X_train.jbl')
        y = joblib.load('output/splits/y_train.jbl')

        gauss_classifier = GaussianNB()
        bernu_classifier = BernoulliNB()

        gauss_classifier.fit(x, y)
        bernu_classifier.fit(x, y)

        joblib.dump(gauss_classifier, 'output/model.jbl')
        joblib.dump(bernu_classifier, 'output/model-bernu.jbl')

        live.log_artifact("output/model.jbl", type="model")
        live.log_artifact("output/model-bernu.jbl", type="model")


if __name__ == "__main__":
    train()

"""Train model(s)"""

import joblib
from dvclive import Live
from sklearn.naive_bayes import GaussianNB


def train():
    """Run training step"""
    with Live() as live:
        live.log_param("classifier", "GaussianNB")
        
        x = joblib.load('output/splits/X_train.jbl')
        y = joblib.load('output/splits/y_train.jbl')

        classifier = GaussianNB()
        classifier.fit(x, y)

        joblib.dump(classifier, 'output/model.jbl')

        live.log_artifact("output/model.jbl", type="model")


if __name__ == "__main__":
    train()

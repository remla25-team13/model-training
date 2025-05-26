"""Train model(s)"""

import joblib

from sklearn.naive_bayes import GaussianNB

def train(x, y):
    """Run training step"""
    classifier = GaussianNB()
    classifier.fit(x, y)

    joblib.dump(classifier, 'output/model.jbl')

if __name__ == "__main__":
    X_train = joblib.load('output/splits/X_train.jbl')
    y_train = joblib.load('output/splits/y_train.jbl')

    train(X_train, y_train)

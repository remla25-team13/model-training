"""Train model(s)"""

import joblib

from sklearn.naive_bayes import GaussianNB

def train():
    """Run training step"""
    x = joblib.load('output/splits/X_train.jbl')
    y = joblib.load('output/splits/y_train.jbl')
    
    classifier = GaussianNB()
    classifier.fit(x, y)

    joblib.dump(classifier, 'output/model.jbl')

if __name__ == "__main__":
    train()

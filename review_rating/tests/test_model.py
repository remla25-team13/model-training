"""
Unit tests for model training and evaluation.

Functions:
    test_model_trainability(split_data):
        Tests if the GaussianNB model can be trained on the provided training data.
        Asserts that the trained model has the 'class_prior_' attribute, indicating successful fitting.

    test_model_accuracy(split_data, classifier):
        Evaluates the accuracy of the provided classifier on the test data.
        Asserts that the accuracy score is greater than 0.6.
"""

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


def test_model_trainability(split_data):
    """Test if the model can be trained."""
    x_train, _, y_train, _ = split_data
    model = GaussianNB()
    model.fit(x_train, y_train)
    assert hasattr(model, "class_prior_")


def test_model_accuracy(split_data, classifier):
    """Test the accuracy of the classifier on the test set."""
    _, X_test, _, y_test = split_data
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.6

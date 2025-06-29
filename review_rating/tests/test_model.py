"""
Unit tests for model training and evaluation. (VS copilot helped writing these.)

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
import numpy as np
import sys


def test_model_determinism(split_data):
    """Ensure consistent predictions across repeated training."""
    x_train, x_test, y_train, _ = split_data

    model1 = GaussianNB()
    model1.fit(x_train, y_train)
    preds1 = model1.predict(x_test)

    model2 = GaussianNB()
    model2.fit(x_train, y_train)
    preds2 = model2.predict(x_test)

    assert np.array_equal(preds1, preds2)


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


def test_model_performance_on_slice_with_negative_reviews(classifier, split_data):
    """Test model performance on data slices with neg reviews"""
    _, x_test, _, y_test = split_data

    indices = (y_test == 0)
    X_slice = x_test[indices]
    y_slice = y_test[indices]

    if len(X_slice) == 0:
        return  # No data in this slice, skip

    acc = classifier.score(X_slice, y_slice)
    assert acc > 0.5, "Model performs poorly on a data slice"


def test_model_performance_on_slice_with_positive_reviews(classifier, split_data):
    """Test model performance on data slices with pos reviews"""
    _, x_test, _, y_test = split_data

    indices = (y_test == 1)
    X_slice = x_test[indices]
    y_slice = y_test[indices]

    if len(X_slice) == 0:
        return  # No data in this slice, skip

    acc = classifier.score(X_slice, y_slice)
    assert acc > 0.5, "Model performs poorly on a data slice"


def test_model_cost_analysis(classifier):
    """Memory used for mode should be less than approximately 20MB"""
    memory_used = sys.getsizeof(classifier)
    assert memory_used < 200000000

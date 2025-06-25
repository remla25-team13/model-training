"""
This module contains tests for monitoring model performance and data drift. (VS copilot helped writing these comments.)
Functions:
    test_prediction_drift(split_data, classifier):
        Tests whether the classifier's prediction accuracy on the test set
        deviates significantly from a predefined baseline accuracy.
    test_data_distribution_drift(split_data):
        Tests whether the mean feature values of the training and test sets
        differ significantly, indicating potential data distribution drift.
"""


import numpy as np
from sklearn.metrics import accuracy_score
import time


def test_prediction_time(classifier, split_data, max_time=0.1):
    """Ensure predictions complete within 100ms per sample."""
    _, X_test, _, _ = split_data
    start = time.time()
    classifier.predict(X_test)
    duration = time.time() - start
    time_per_sample = duration / len(X_test)
    assert time_per_sample < max_time


def test_prediction_drift(split_data, classifier):
    """
    Test to detect prediction drift by comparing the classifier's accuracy on the test set to a predefined baseline.
    Args:
        split_data (tuple): A tuple containing the training and test data splits (X_train, X_test, y_train, y_test).
        classifier (sklearn.base.BaseEstimator): The trained classifier to be evaluated.
    Asserts:
        The absolute difference between the current accuracy and the baseline accuracy (0.9) is less than 0.15,
        indicating no significant prediction drift has occurred.
    """
    _, x_test, _, y_test = split_data
    y_pred = classifier.predict(x_test)
    baseline = 0.9
    accuracy = accuracy_score(y_test, y_pred)
    assert abs(accuracy - baseline) < 0.15


def test_data_distribution_drift(split_data):
    """
    Test to detect data distribution drift between training and test datasets.
    This test calculates the mean of each feature in both the training and test sets,
    computes the absolute difference (drift) between these means, and asserts that the
    average drift across all features is below a specified threshold (0.1). This helps
    ensure that the test data distribution is similar to the training data distribution,
    which is important for model generalization.
    Args:
        split_data (tuple): A tuple containing the training and test feature sets (X_train, X_test)
            and their corresponding labels (ignored in this test).
    Raises:
        AssertionError: If the mean drift between training and test data exceeds 0.1.
    """
    x_train, x_test, _, _ = split_data
    train_mean = np.mean(x_train, axis=0)
    test_mean = np.mean(x_test, axis=0)
    drift = np.abs(train_mean - test_mean)
    assert np.mean(drift) < 0.1


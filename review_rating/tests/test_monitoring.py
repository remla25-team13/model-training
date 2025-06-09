"""
Tests for monitoring model performance and data drift.
"""

import numpy as np
from sklearn.metrics import accuracy_score


def test_prediction_drift(split_data, classifier):
    """
    Test if classifier accuracy deviates from baseline on test set.
    """
    _, X_test, _, y_test = split_data
    y_pred = classifier.predict(X_test)
    baseline = 0.9
    accuracy = accuracy_score(y_test, y_pred)
    assert abs(accuracy - baseline) < 0.15


def test_data_distribution_drift(split_data):
    """
    Test whether feature means differ too much between train and test sets.
    """
    X_train, X_test, _, _ = split_data
    train_mean = np.mean(X_train, axis=0)
    test_mean = np.mean(X_test, axis=0)
    drift = np.abs(train_mean - test_mean)
    assert np.mean(drift) < 0.1

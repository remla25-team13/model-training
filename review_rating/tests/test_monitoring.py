"""
Tests for monitoring model performance and data drift.
"""

import numpy as np
from sklearn.metrics import accuracy_score

def test_prediction_drift(split_data, classifier):
    """Test classifier accuracy is close to baseline."""
    _, x_test, _, y_test = split_data
    y_pred = classifier.predict(x_test)
    baseline = 0.9
    accuracy = accuracy_score(y_test, y_pred)
    assert abs(accuracy - baseline) < 0.15

def test_data_distribution_drift(split_data):
    """Test feature distribution consistency between train/test."""
    x_train, x_test, _, _ = split_data
    train_mean = np.mean(x_train, axis=0)
    test_mean = np.mean(x_test, axis=0)
    drift = np.abs(train_mean - test_mean)
    assert np.mean(drift) < 0.1

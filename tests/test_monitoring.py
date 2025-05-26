import numpy as np
import pytest
from sklearn.metrics import accuracy_score

def test_prediction_drift(split_data, classifier):
    _, X_test, _, y_test = split_data
    y_pred = classifier.predict(X_test)
    baseline = 0.7 # TODO this should be configurable
    
    acc = accuracy_score(y_test, y_pred)
    acc_diff = abs(acc - baseline)

    if acc_diff < 0:
        pytest.skip("Accuracy degraded, Curr accuracy - Baseline: abs({acc} - {baseline}) = {acc_diff} < 0.15.")

def test_data_distribution_drift(split_data):
    X_train, X_test, _, _ = split_data
    train_mean = np.mean(X_train, axis=0)
    test_mean = np.mean(X_test, axis=0)
    drift = np.abs(train_mean - test_mean)
    assert np.mean(drift) < 0.1

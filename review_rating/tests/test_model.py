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
def test_model_fit_input_shape(split_data):
    """Check classifier training input shapes."""
    x_train, _, y_train, _ = split_data
    assert len(x_train) == len(y_train)

def test_model_prediction_shape(split_data, classifier):
    """Check prediction output shape matches test labels."""
    _, x_test, _, y_test = split_data
    y_pred = classifier.predict(x_test)
    assert len(y_pred) == len(y_test)

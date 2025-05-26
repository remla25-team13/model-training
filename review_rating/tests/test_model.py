from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def test_model_trainability(split_data):
    X_train, _, y_train, _ = split_data
    model = GaussianNB()
    model.fit(X_train, y_train)
    assert hasattr(model, "class_prior_")


def test_model_accuracy(split_data, classifier):
    _, X_test, _, y_test = split_data
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.6

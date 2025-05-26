import os
import pickle

import joblib


def test_model_serialized():
    assert os.path.exists("sentiment_model.pkl")
    model = joblib.load("sentiment_model.pkl")
    assert hasattr(model, "predict")


def test_vectorizer_serialized():
    assert os.path.exists("sentiment_model_vectorizer.pkl")
    vec = pickle.load(open("sentiment_model_vectorizer.pkl", "rb"))
    assert hasattr(vec, "transform")


def test_prediction_pipeline(preprocessor):
    text = "This place was not great"
    vec = pickle.load(open("sentiment_model_vectorizer.pkl", "rb"))
    model = joblib.load("sentiment_model.pkl")

    processed = preprocessor.preprocess(text)
    X = vec.transform([processed]).toarray()
    pred = model.predict(X)
    assert pred[0] in [0, 1]

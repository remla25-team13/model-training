"""Test the infrastructure of the review rating package."""

import os
import joblib


MODEL_PATH_A = "processed/model-gauss.jbl"
MODEL_PATH_B = "processed/model-multi.jbl"
VECTORIZER_PATH = "processed/vectorizer.jbl"


def test_model_serialized():
    """Check that the models are serialized and can be loaded."""
    for model_path in [MODEL_PATH_A, MODEL_PATH_B]:
        assert os.path.exists(model_path)
        model = joblib.load(model_path)
        assert hasattr(model, "predict")


def test_vectorizer_serialized():
    """Check that the vectorizer is serialized and can be loaded."""
    assert os.path.exists("processed/vectorizer.pkl")
    vec = joblib.load("processed/vectorizer.pkl")
    assert hasattr(vec, "transform")


def test_prediction_pipeline(preprocessor):
    """Test the end-to-end prediction pipeline."""
    text = "This place was not great"
    vec = joblib.load("processed/vectorizer.pkl")

    for model_path in [MODEL_PATH_A, MODEL_PATH_B]:
        model = joblib.load(model_path)

        processed = preprocessor.preprocess(text)
        X = vec.transform([processed]).toarray()
        pred = model.predict(X)
        assert pred[0] in [0, 1]

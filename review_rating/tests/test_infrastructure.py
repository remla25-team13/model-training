"""
Infrastructure Tests for Serialized Models and Vectorizer.

This module tests whether trained models and vectorizer are correctly saved to disk
and whether they can make predictions on new input text after loading.
"""

import os
import joblib


MODEL_PATH_A = "processed/model-gauss.jbl"
MODEL_PATH_B = "processed/model-multi.jbl"
VECTORIZER_PATH = "processed/vectorizer.pkl"


def test_model_serialized():
    """Check that models are saved and implement predict()."""
    for model_path in [MODEL_PATH_A, MODEL_PATH_B]:
        assert os.path.exists(model_path)
        model = joblib.load(model_path)
        assert hasattr(model, "predict")


def test_vectorizer_serialized():
    """Check that the vectorizer is saved and implements transform()."""
    assert os.path.exists(VECTORIZER_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    assert hasattr(vectorizer, "transform")


def test_prediction_pipeline(text_preprocessor):
    """Test end-to-end prediction using serialized model and vectorizer."""
    text = "This place was not great"
    vectorizer = joblib.load(VECTORIZER_PATH)

    for model_path in [MODEL_PATH_A, MODEL_PATH_B]:
        model = joblib.load(model_path)
        processed = text_preprocessor.preprocess(text)
        features = vectorizer.transform([processed]).toarray()
        prediction = model.predict(features)
        assert prediction[0] in [0, 1]

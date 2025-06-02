import os
import pickle  # nosec

import joblib


MODEL_PATH_A = "processed/model-gauss.jbl"
MODEL_PATH_B = "processed/model-multi.jbl"
VECTORIZER_PATH = "processed/vectorizer.jbl"


def test_model_serialized():
    for model_path in [MODEL_PATH_A, MODEL_PATH_B]:
        assert os.path.exists(model_path)
        model = joblib.load(model_path)
        assert hasattr(model, "predict")


def test_vectorizer_serialized():
    assert os.path.exists("processed/vectorizer.pkl")
    vec = pickle.load(open("processed/vectorizer.pkl", "rb"))  # nosec
    assert hasattr(vec, "transform")


def test_prediction_pipeline(preprocessor):
    text = "This place was not great"
    vec = pickle.load(open("processed/vectorizer.pkl", "rb"))  # nosec

    for model_path in [MODEL_PATH_A, MODEL_PATH_B]:
        model = joblib.load(model_path)

        processed = preprocessor.preprocess(text)
        X = vec.transform([processed]).toarray()
        pred = model.predict(X)
        assert pred[0] in [0, 1]

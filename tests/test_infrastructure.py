import os
import joblib

MODEL_PATH = 'output/model.jbl'
VECTORIZER_PATH = 'output/vectorizer.jbl'

def test_model_serialized():
    assert os.path.exists(MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, 'predict')

def test_vectorizer_serialized():
    assert os.path.exists(VECTORIZER_PATH)
    vec = joblib.load(open(VECTORIZER_PATH, 'rb'))
    assert hasattr(vec, 'transform')

def test_prediction_pipeline(preprocessor):
    text = "This place was not great"
    vec = joblib.load(open(VECTORIZER_PATH, 'rb'))
    model = joblib.load(MODEL_PATH)

    processed = preprocessor.preprocess(text)
    X = vec.transform([processed]).toarray()
    pred = model.predict(X)
    assert pred[0] in [0, 1]

import os
import joblib

MODEL_PATH_A = 'output/model-gauss.jbl'
MODEL_PATH_B = 'output/model-multi.jbl'
VECTORIZER_PATH = 'output/vectorizer.jbl'

def test_model_serialized():
    assert os.path.exists(MODEL_PATH_A)
    assert os.path.exists(MODEL_PATH_B)
    model = joblib.load(MODEL_PATH_A)
    model = joblib.load(MODEL_PATH_B)
    assert hasattr(model, 'predict')

def test_vectorizer_serialized():
    assert os.path.exists(VECTORIZER_PATH)
    vec = joblib.load(open(VECTORIZER_PATH, 'rb'))
    assert hasattr(vec, 'transform')

def test_prediction_pipeline(preprocessor):
    text = "This place was not great"
    vec = joblib.load(open(VECTORIZER_PATH, 'rb'))
    model_a = joblib.load(MODEL_PATH_A)
    model_b = joblib.load(MODEL_PATH_B)

    processed = preprocessor.preprocess(text)
    X = vec.transform([processed]).toarray()
    pred_a = model_a.predict(X)
    pred_b = model_b.predict(X)
    assert pred_a[0] in [0, 1] and pred_b[0] in [0, 1]

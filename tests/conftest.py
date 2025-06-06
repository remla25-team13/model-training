import pytest
import pandas as pd
import os
import joblib
from lib_ml import Preprocessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from training.get_data import get_data
from training.preprocessing import preprocess
from training.build_vectorizer import build_vectorizer
from training.train_model import train
from training.get_metrics import get_metrics

MODEL_PATH_A = "output/model-gauss.jbl"
MODEL_PATH_B = "output/model-multi.jbl"
VECTORIZER_PATH = "output/vectorizer.jbl"
DATASET_PATH = "output/reviews.tsv"

@pytest.fixture(scope="session", autouse=True)
def build_artifacts():
    """Run training script once per test session to generate model/vectorizer."""
    
    get_data()
    preprocess()
    build_vectorizer()
    train()
    get_metrics()

    assert os.path.exists(MODEL_PATH_A), "Model A file not created"
    assert os.path.exists(MODEL_PATH_B), "Model B file not created"
    assert os.path.exists(VECTORIZER_PATH), "Vectorizer file not created"

@pytest.fixture(scope="session")
def dataset():
    return pd.read_csv(DATASET_PATH, delimiter='\t', quoting=3)

@pytest.fixture(scope="session")
def preprocessor():
    return Preprocessor()

@pytest.fixture(scope="session")
def corpus(dataset, preprocessor):
    return [preprocessor.preprocess(review) for review in dataset['Review'][:900]]

@pytest.fixture(scope="session")
def vectorizer(corpus):
    cv = CountVectorizer(max_features=1420)
    cv.fit(corpus)
    return cv

@pytest.fixture(scope="session")
def X_y(corpus, dataset, vectorizer):
    X = vectorizer.transform(corpus).toarray()
    y = dataset.iloc[:900, -1].values
    return X, y

@pytest.fixture(scope="session")
def split_data(X_y):
    X, y = X_y
    return train_test_split(X, y, test_size=0.2, random_state=0)

@pytest.fixture(scope="session")
def classifier(split_data):
    clf = joblib.load(MODEL_PATH_A)
    clf = joblib.load(MODEL_PATH_B)
    return clf

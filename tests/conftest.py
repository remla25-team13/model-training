import pytest
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import subprocess
import pickle
import joblib
from lib_ml import Preprocessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from training import model_analysis


MODEL_PATH = "sentiment_model.pk1"
VECTORIZER_PATH = "bow_vectorizer.pkl"

@pytest.fixture(scope="session")
def build_artifacts():
    """Run training script once per test session to generate model/vectorizer."""
    model_analysis.run_pipeline()
    assert os.path.exists(MODEL_PATH), "Model file not created"
    assert os.path.exists(VECTORIZER_PATH), "Vectorizer file not created"

@pytest.fixture(scope="session")
def dataset():
    return pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)

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
    X_train, _, y_train, _ = split_data
    clf = joblib.load('sentiment_model.pk1')
    return clf

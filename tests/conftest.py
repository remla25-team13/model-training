import pytest
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import joblib
from lib_ml import Preprocessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

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

"""
Pytest fixtures for training, preprocessing, and data loading.
"""
import os
import joblib
import pandas as pd
import pytest
from lib_ml import Preprocessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from review_rating.modeling.prepare_data import prepare_data
from review_rating.modeling.train import train_model

MODEL_PATH_A = "processed/model-gauss.jbl"
MODEL_PATH_B = "processed/model-multi.jbl"
VECTORIZER_PATH = "processed/vectorizer.pkl"
DATASET_PATH = "output/reviews-latest.tsv"


@pytest.fixture(scope="session", autouse=True)
def build_artifacts():
    """Run training script once per test session to generate model/vectorizer."""
    prepare_data(
        input_path="output/reviews.tsv",
        output_dir="processed",
        test_split=0.2,
        random_state=42,
        max_features=1420,
    )

    assert os.path.exists("processed/processed_data.pk1"), "Data file not created"
    assert os.path.exists("processed/vectorizer.pkl"), "Vectorizer file not created"

    train_model(
        input_dir="processed",
        output_path="processed/",
    )

    assert os.path.exists(MODEL_PATH_A), "Model A file not created"
    assert os.path.exists(MODEL_PATH_B), "Model B file not created"
    assert os.path.exists(VECTORIZER_PATH), "Vectorizer file not created"


@pytest.fixture(scope="session")
def dataset():
    """Load the dataset from the TSV file."""
    return pd.read_csv(
        DATASET_PATH, delimiter="\t", quoting=3
    )


@pytest.fixture(scope="session")
def preprocessor():
    """Create a preprocessor instance."""
    return Preprocessor()


@pytest.fixture(scope="session")
def corpus(dataset, preprocessor):
    """Preprocess the review text data."""
    return [preprocessor.preprocess(review) for review in dataset["Review"][:900]]


@pytest.fixture(scope="session")
def vectorizer(corpus):
    """Create and fit a CountVectorizer on the corpus."""
    cv = CountVectorizer(max_features=1420)
    cv.fit(corpus)
    return cv


@pytest.fixture(scope="session")
def X_y(corpus, dataset, vectorizer):
    """Transform the corpus into feature vectors and extract labels."""
    X = vectorizer.transform(corpus).toarray()
    y = dataset.iloc[:900, -1].to_numpy()
    return X, y


@pytest.fixture(scope="session")
def split_data(X_y):
    """Split the data into training and testing sets."""
    X, y = X_y
    return train_test_split(X, y, test_size=0.2, random_state=0)


@pytest.fixture(scope="session")
def classifier():
    """Load the trained classifiers."""
    clf = joblib.load(MODEL_PATH_A)
    clf = joblib.load(MODEL_PATH_B)
    return clf

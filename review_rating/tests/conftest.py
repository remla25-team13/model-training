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
DATASET_PATH = "output/reviews.tsv"


@pytest.fixture(scope="session", autouse=True)
def build_artifacts():
    """Run training script once per test session to generate model/vectorizer."""
    prepare_data(
        input_path="data/raw/a1_RestaurantReviews_HistoricDump.tsv",
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
    """Load raw dataset for testing."""
    return pd.read_csv(
        "data/raw/a1_RestaurantReviews_HistoricDump.tsv", delimiter="\t", quoting=3
    )


@pytest.fixture(scope="session")
def text_preprocessor():
    """Create a Preprocessor instance."""
    return Preprocessor()


@pytest.fixture(scope="session")
def review_corpus(raw_dataset, preproc):
    """Preprocess reviews into a corpus."""
    return [preproc.preprocess(r) for r in raw_dataset["Review"][:900]]


@pytest.fixture(scope="session")
def vectorizer(preprocessed_reviews):
    """Fit CountVectorizer to corpus."""
    cv = CountVectorizer(max_features=1420)
    cv.fit(preprocessed_reviews)
    return cv


@pytest.fixture(scope="session")
def feat_label_data(preproc_reviews, raw_dataset, text_vectorizer):
    """Return features and labels."""
    features = text_vectorizer.transform(preproc_reviews).toarray()
    labels = raw_dataset.iloc[:900, -1].to_numpy()
    return features, labels


@pytest.fixture(scope="session")
def split_data(feat_label_data):
    """Split data into train/test sets."""
    features, labels = feat_label_data
    return train_test_split(features, labels, test_size=0.2, random_state=0)


@pytest.fixture(scope="session")
def classifier():
    """Load trained classifier."""
    return joblib.load(MODEL_PATH_B)

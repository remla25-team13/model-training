"""Train and evaluate a Naive Bayes sentiment model for restaurant reviews with separate stages."""

import pickle
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from lib_ml import Preprocessor
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def load_data(filepath: str = "a1_RestaurantReviews_HistoricDump.tsv") -> pd.DataFrame:
    """Load dataset from TSV file."""
    dataset = pd.read_csv(filepath, delimiter="\t", quoting=3)
    logger.info(
        f"Loaded {len(dataset)} rows and {len(dataset.columns)} columns from the TSV file."
    )
    return dataset


def prepare_data(
    input_path: str,
    output_dir: str,
    test_split: float = 0.2,
    random_state: int = 42,
    max_features: int = 1420,
) -> None:
    """
    Prepare and split the data, saving processed data and vectorizer.

    Args:
        input_path: Path to the input TSV file
        output_dir: Directory to save processed data
        test_split: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        max_features: Maximum number of features for vectorizer
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    dataset = load_data(filepath=input_path)
    reviews = dataset["Review"]
    ratings = dataset["Liked"]

    # Preprocess text and vectorize
    corpus = [Preprocessor().preprocess(review) for review in reviews]
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus).toarray()
    y = ratings.values

    # Save vectorizer
    vectorizer_path = f"{output_dir}/vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Saved vectorizer to {vectorizer_path}")

    # Split data
    if test_split == 0.0:
        X_train, X_test, y_train, y_test = X, [], y, []
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_state
        )

    # Save processed data
    data_path = f"{output_dir}/processed_data.pk1"
    joblib.dump((X_train, X_test, y_train, y_test), data_path)
    logger.info(
        f"Split the data into {len(X_train)} training and {len(X_test)} testing samples."
    )
    logger.info(f"Saved processed data to {data_path}")


def train_model(
    input_dir: str,
    output_model: str,
) -> GaussianNB:
    """
    Train a model using prepared data and save it.

    Args:
        input_dir: Directory containing processed data
        output_model: Path to save the trained model
        random_state: Random seed for reproducibility

    Returns:
        Trained GaussianNB classifier
    """
    # Load processed data
    data_path = f"{input_dir}/processed_data.pk1"
    X_train, _, y_train, _ = joblib.load(data_path)
    logger.info(f"Loaded training data with {len(X_train)} samples from {data_path}")

    # Train model
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    logger.info("Model training completed")

    # Save model
    model_dir = Path(output_model).parent
    if model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, output_model)
    logger.info(f"Saved trained model to {output_model}")

    return classifier


def evaluate_model(
    input_dir: str,
    model_path: str,
) -> None:
    """
    Evaluate a trained model using test data and save results.

    Args:
        input_dir: Directory containing processed data
        model_path: Path to the trained model
        random_state: Random seed for reproducibility
    """
    # Load data and model
    data_path = f"{input_dir}/processed_data.pk1"
    _, X_test, _, y_test = joblib.load(data_path)

    classifier = joblib.load(model_path)
    if len(X_test) == 0 or len(y_test) == 0:
        logger.warning("No test data available. Skipping evaluation.")
        return

    # Evaluate model
    y_pred = classifier.predict(X_test)

    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")


def load_data_and_model(
    input_dir: str,
    model_path: str,
) -> Tuple[Tuple, GaussianNB]:
    """
    Helper function to load processed data and trained model.

    Args:
        input_dir: Directory containing processed data
        model_path: Path to the trained model

    Returns:
        Tuple of (data, model) where data is (X_train, X_test, y_train, y_test)
    """
    data = joblib.load(f"{input_dir}/processed_data.pk1")
    model = joblib.load(model_path)
    return data, model

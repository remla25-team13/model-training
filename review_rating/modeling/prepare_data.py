import pickle
from pathlib import Path

import joblib
from lib_ml import Preprocessor
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from review_rating.modeling import load_data


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

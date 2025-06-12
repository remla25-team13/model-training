"""
This module provides functionality to prepare and split review data for model training.

It includes:
- Loading review data from a TSV file.
- Preprocessing review text using a custom Preprocessor.
- Vectorizing the preprocessed text using CountVectorizer with a configurable maximum number of features.
- Splitting the data into training and testing sets.
- Saving the fitted vectorizer and processed data to disk for later use.
"""

from pathlib import Path

import joblib
from lib_ml import Preprocessor
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from review_rating.modeling import load_data  # nosec


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

    print(dataset)

    # Preprocess text and vectorize
    vectorizer = CountVectorizer(max_features=max_features)
    x = vectorizer.fit_transform(
        [Preprocessor().preprocess(review) for review in dataset["Review"]]
    ).toarray()
    y = dataset["Liked"].to_numpy()

    # Save vectorizer
    vectorizer_path = f"{output_dir}/vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    logger.info(f"Saved vectorizer to {vectorizer_path}")

    # Split data
    if test_split == 0.0:
        x_train, x_test, y_train, y_test = x, [], y, []
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_split, random_state=random_state
        )

    # Save processed data
    joblib.dump((x_train, x_test, y_train, y_test), f"{output_dir}/processed_data.pk1")
    logger.info(
        f"Split the data into {len(x_train)} training and {len(x_test)} testing samples."
    )
    logger.info(f"Saved processed data to {f"{output_dir}/processed_data.pk1"}")

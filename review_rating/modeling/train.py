"""Main entry point for training a sentiment analysis model on restaurant reviews."""

from pathlib import Path

import joblib
from loguru import logger
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from dvclive import Live


def train_model(
    input_dir: str, output_path: str, live_logger: Live | None = None
) -> GaussianNB:
    """
    Train a model using prepared data and save it.

    Args:
        input_dir: Directory containing processed data
        output_path: Path to save the trained model
        random_state: Random seed for reproducibility

    Returns:
        Trained GaussianNB classifier
    """

    if live_logger is not None:
        live_logger.log_param("classifiers", "GaussianNB, BernouliNB")

    # Load processed data
    data_path = f"{input_dir}/processed_data.pk1"
    x_train, _, y_train, _ = joblib.load(data_path)
    logger.info(f"Loaded training data with {len(x_train)} samples from {data_path}")

    # Train model
    gauss_classifier = GaussianNB()
    multi_classifier = MultinomialNB()
    gauss_classifier.fit(x_train, y_train)
    multi_classifier.fit(x_train, y_train)
    logger.info("Model training completed")

    # Save model
    Path(output_path).mkdir(parents=True, exist_ok=True)

    joblib.dump(gauss_classifier, f"{output_path}/model-gauss.jbl")
    joblib.dump(multi_classifier, f"{output_path}/model-multi.jbl")
    logger.info(f"Saved trained models to {output_path}")

    if live_logger is not None:
        live_logger.log_artifact(f"{output_path}/model-gauss.jbl", type="model")
        live_logger.log_artifact(f"{output_path}/model-multi.jbl", type="model")

    return gauss_classifier, multi_classifier

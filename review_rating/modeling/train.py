"""Main entry point for training a sentiment analysis model on restaurant reviews."""
from pathlib import Path

import joblib
from loguru import logger
from sklearn.naive_bayes import GaussianNB


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
    x_train, _, y_train, _ = joblib.load(data_path)
    logger.info(f"Loaded training data with {len(x_train)} samples from {data_path}")

    # Train model
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    logger.info("Model training completed")

    # Save model
    model_dir = Path(output_model).parent
    if model_dir:
        model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, output_model)
    logger.info(f"Saved trained model to {output_model}")

    return classifier

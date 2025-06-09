"""Main entry point for evaluating a sentiment analysis model on restaurant reviews."""

import json
from datetime import datetime

import joblib
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix

models = ["gauss", "multi"]


def evaluate_model(
    input_dir: str,
    model_path: str,
    metrics_path: str,
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
    _, x_test, _, y_test = joblib.load(data_path)

    if len(x_test) == 0 or len(y_test) == 0:
        logger.warning("No test data available. Skipping evaluation.")
        return

    metrics_obj = {}
    for model_type in models:
        classifier = joblib.load(f"{model_path}/model-{model_type}.jbl")
        logger.info(
            f"Loaded {model_type} model from {model_path}/model-{model_type}.jbl"
        )

        # Evaluate model
        y_pred = classifier.predict(x_test)

        today = datetime.today().isoformat()
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        metrics_obj[model_type] = {
            "date": f"{today}",
            "confusion_matrix": f"{cm}",
            "accuracy": f"{accuracy}",
        }

    print(json.dumps(metrics_obj, indent=2))
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_obj, f, indent=2)

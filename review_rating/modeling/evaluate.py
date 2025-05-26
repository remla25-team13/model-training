import joblib
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix


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

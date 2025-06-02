"""Calculate model performance"""
from datetime import datetime
import json

import joblib
from sklearn.metrics import confusion_matrix, accuracy_score

models = ['gauss', 'multi']


def get_metrics():
    """Run metrics step"""
    x = joblib.load('output/splits/X_test.jbl')
    y = joblib.load('output/splits/y_test.jbl')
    metrics_obj = {
        "gauss": {},
        "multi": {}
    }

    for type in models:
        cls = joblib.load(f'output/model-{type}.jbl')
        y_pred = cls.predict(x)

        today = datetime.today().isoformat()
        cm = confusion_matrix(y, y_pred)
        accuracy = accuracy_score(y, y_pred)

        metrics_obj[type] = {
            "date": f"{today}",
            "confusion_matrix": f"{cm}",
            "accuracy": f"{accuracy}"
        }

    with open('output/metrics.json', 'w', encoding="utf-8") as f:
        json.dump(metrics_obj, f)


if __name__ == "__main__":
    get_metrics()

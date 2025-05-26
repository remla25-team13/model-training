"""Calculate model performance"""
from datetime import datetime
import json

import joblib
from sklearn.metrics import confusion_matrix, accuracy_score

def get_metrics():
    """Run metrics step"""
    cls=joblib.load('output/model.jbl')
    x = joblib.load('output/splits/X_test.jbl')
    y = joblib.load('output/splits/y_test.jbl')
    
    y_pred = cls.predict(x)

    today = datetime.today().isoformat()
    cm = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)

    with open('output/metrics.json', 'w', encoding="utf-8") as f:
        metrics_obj = {
            "date": f"{today}",
            "confusion_matrix": f"{cm}",
            "accuracy": f"{accuracy}"
        }

        json.dump(metrics_obj, f)

if __name__ == "__main__":
    get_metrics()

"""Calculate model performance"""
from datetime import datetime

import joblib
from sklearn.metrics import confusion_matrix, accuracy_score

classifier = joblib.load('output/model.jbl')
X_test = joblib.load('output/splits/X_test.jbl')
y_test = joblib.load('output/splits/y_test.jbl')

y_pred = classifier.predict(X_test)

today = datetime.today().isoformat()
cm = confusion_matrix(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)

with open('output/metrics.txt', 'w', encoding="utf-8") as f:
    f.write(f"Date: {today} \nConfussion Matrix:\n{cm} \nAccuracy score: {accuracy_score:.2f}")

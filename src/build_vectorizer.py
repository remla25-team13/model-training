import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

TEST_SIZE = float(os.getenv('TEST_SIZE'))
RNG_STATE = int(os.getenv('RNG_STATE'))
MAX_FEATURES = int(os.getenv('VECTORIZER_MAX_FEATURES'))
y = joblib.load('output/labels.jbl')

corpus = joblib.load('output/corpus.jbl')
cv = CountVectorizer(max_features=MAX_FEATURES)

X = cv.fit_transform(corpus).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RNG_STATE)

joblib.dump(cv, 'output/vectorizer.jbl')
joblib.dump(X_train, 'output/splits/X_train.jbl')
joblib.dump(X_test, 'output/splits/X_test.jbl')
joblib.dump(y_train, 'output/splits/y_train.jbl')
joblib.dump(y_test, 'output/splits/y_test.jbl')

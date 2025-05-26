"""Build vectorizer"""

import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def build_vectorizer(test_size=0.2, rng_state=42, max_features=1420):
    """Run vectorization step"""
    labels=joblib.load('output/labels.jbl')
    corpus=joblib.load('output/corpus.jbl')

    cv = CountVectorizer(max_features=max_features)

    x_vectorized = cv.fit_transform(corpus).toarray()
    x_train, x_test, y_train, y_test = train_test_split(x_vectorized, labels,
        test_size=test_size, random_state=rng_state)

    joblib.dump(cv, 'output/vectorizer.jbl')
    joblib.dump(x_train, 'output/splits/X_train.jbl')
    joblib.dump(x_test, 'output/splits/X_test.jbl')
    joblib.dump(y_train, 'output/splits/y_train.jbl')
    joblib.dump(y_test, 'output/splits/y_test.jbl')


if __name__ == '__main__':
    TEST_SIZE = float(os.getenv('TEST_SIZE', '0.2'))
    RNG_STATE = int(os.getenv('RNG_STATE', '42'))
    MAX_FEATURES = int(os.getenv('VECTORIZER_MAX_FEATURES', '1420'))

    build_vectorizer(TEST_SIZE, RNG_STATE, MAX_FEATURES)

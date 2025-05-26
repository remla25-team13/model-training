"""Train and evaluate a Naive Bayes sentiment model for restaurant reviews."""

import pickle # nosec B403
import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from lib_ml import Preprocessor


def load_data(filepath='a1_RestaurantReviews_HistoricDump.tsv'):
    '''Load dataset and stopwords'''
    dataset = pd.read_csv(filepath, delimiter='\t', quoting=3)
    return dataset


def preprocess_data(dataset, max_features=1420):
    '''Preprocess data'''
    corpus = []
    for i in range(0, 900):
        review = dataset['Review'][i]
        review = Preprocessor().preprocess(review)
        corpus.append(review)

    vectorizer = CountVectorizer(max_features=max_features)
    x = vectorizer.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values  # Match indices used in corpus

    with open('bow_vectorizer.pkl', "wb") as f:
        pickle.dump(vectorizer, f)

    return x, y


def train_model(x, y):
    '''Function which handles thhe training of the model'''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)

    joblib.dump(classifier, 'sentiment_model.pk1')
    return classifier, x_test, y_test


def evaluate_model(classifier, x_test, y_test):
    '''Function to evaluate a classifier'''
    y_pred = classifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))


def run_pipeline():
    '''Function to run a model training pipeline'''
    dataset = load_data()
    x, y = preprocess_data(dataset)
    classifier, x_test, y_test = train_model(x, y)
    evaluate_model(classifier, x_test, y_test)


if __name__ == "__main__":
    run_pipeline()

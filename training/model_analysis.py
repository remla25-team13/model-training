"""Train and evaluate a Naive Bayes sentiment model for restaurant reviews."""

import pickle
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from lib_ml import Preprocessor

def run_pipeline():
    '''Function to run amodel training pipeline'''
    # Load dataset and stopwords
    dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)
    nltk.download('stopwords')

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    # Preprocess reviews
    corpus = []
    for i in range(0, 900):
        review = dataset['Review'][i]
        review = Preprocessor().preprocess(review)
        corpus.append(review)

    # Convert text to features
    cv = CountVectorizer(max_features=1420)
    x = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    # Save the BoW vectorizer
    with open('bow_vectorizer.pkl', "wb") as f:
        pickle.dump(cv, f)

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    # Train classifier
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)

    # Save model
    joblib.dump(classifier, 'sentiment_model.pk1')

    # Evaluate
    y_pred = classifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    run_pipeline()

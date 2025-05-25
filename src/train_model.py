import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X = joblib.load('output/splits/X_train.jbl')
y = joblib.load('output/splits/y_train.jbl')

classifier = GaussianNB()
classifier.fit(X, y)

joblib.dump(classifier, 'output/model.jbl')

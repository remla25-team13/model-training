"""Preprocess data"""

import joblib
import pandas as pd

from tqdm import trange

from lib_ml import Preprocessor


def preprocess():
    """Run preprocessing step"""
    dataset = pd.read_csv('output/reviews-latest.tsv', delimiter='\t', quoting=3)
    preprocessor = Preprocessor()

    corpus = []
    labels = []
    for i in trange(len(dataset)):
        review, label = dataset.iloc[i]

        review = preprocessor.preprocess(review)
        corpus.append(review)
        labels.append(label)

    joblib.dump(corpus, "output/corpus.jbl")
    joblib.dump(labels, "output/labels.jbl")


if __name__ == "__main__":
    preprocess()

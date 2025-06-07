""" Download relevant data """

import nltk
from dvc.api import open as dvc_open
import pandas as pd


def get_data():
    """Run data download step"""
    nltk.download('stopwords')

    with dvc_open("output/reviews.tsv", mode='r') as f:
        dataset = pd.read_csv(f, delimiter='\t', quoting=3)

        dataset.to_csv('output/reviews-latest.tsv', sep="\t", quoting=3, index=False)

        print("Saved latest reviews version from DVC")


if __name__ == '__main__':
    get_data()

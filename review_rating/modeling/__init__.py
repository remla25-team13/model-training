"""Model training package"""

from loguru import logger
import nltk
import pandas as pd
from dvc.api import open as dvc_open


def load_data(filepath: str = "output/reviews.tsv") -> pd.DataFrame:
    """Load dataset from TSV file."""
    nltk.download("stopwords")

    with dvc_open(filepath, mode='r') as f:
        dataset = pd.read_csv(f, delimiter="\t", quoting=3)

        dataset.to_csv("output/reviews-latest.tsv", sep="\t", quoting=3, index=False)

        print("Saved latest reviews version from DVC")

        return dataset

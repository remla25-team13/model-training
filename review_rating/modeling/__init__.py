"""Model training package"""

from loguru import logger
import pandas as pd


def load_data(filepath: str = "a1_RestaurantReviews_HistoricDump.tsv") -> pd.DataFrame:
    """Load dataset from TSV file."""
    dataset = pd.read_csv(filepath, delimiter="\t", quoting=3)
    logger.info(
        f"Loaded {len(dataset)} rows and {len(dataset.columns)} columns from the TSV file."
    )
    return dataset

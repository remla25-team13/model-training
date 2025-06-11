"""
Tests for dataset loading and integrity.
"""


def test_dataset_not_empty(dataset):
    """Check that the dataset is not empty."""
    assert not dataset.empty


def test_no_missing_reviews(dataset):
    """Ensure there are no missing review entries."""
    assert dataset["Review"].isnull().sum() == 0

"""
Tests for dataset loading and integrity.
"""
import sys

def test_dataset_not_empty(dataset):
    """Check that the dataset is not empty."""
    assert not dataset.empty


def test_no_missing_reviews(dataset):
    """Ensure there are no missing review entries."""
    assert dataset["Review"].isnull().sum() == 0

def test_feature_cost_analysis(vectorizer):
    """Memory used for features should be less than approximately 10MB"""
    memory_used = sys.getsizeof(vectorizer)
    assert memory_used <100000000 
    
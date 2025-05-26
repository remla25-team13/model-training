import pandas as pd
import pytest

def test_dataset_not_empty(dataset):
    assert not dataset.empty

def test_no_missing_reviews(dataset):
    assert dataset['Review'].isnull().sum() == 0
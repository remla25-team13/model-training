"""
Mutamorphic tests for sentiment model robustness.

These tests validate that the sentiment model is stable under
contextual changes such as synonym replacements.
"""

import pytest
import joblib
import numpy as np


# Define test cases: (original_text, mutated_text)
# Both texts should ideally have the same sentiment
MUTAMORPHIC_PAIRS = [
    ("The food was okay", "The food was fine"),
    ("The waiter was rude", "The waiter was impolite"),
    ("I loved the ambiance", "I liked the ambiance"),
    ("The service was bad", "The service was terrible"),
    ("It was a great experience", "It was a fantastic experience"),
    ("Not the best dinner", "Not the greatest dinner"),
    ("I would not recommend it", "I wouldn't suggest it"),
    ("Service could be improved", "Service could be better"),
]


@pytest.mark.parametrize("original, mutated", MUTAMORPHIC_PAIRS)
def test_prediction_consistency(original, mutated, preprocessor, vectorizer):
    """Check that synonym-swapped inputs produce the same prediction."""

    processed_orig = preprocessor.preprocess(original)
    processed_mut = preprocessor.preprocess(mutated)

    X_orig = vectorizer.transform([processed_orig]).toarray()
    X_mut = vectorizer.transform([processed_mut]).toarray()

    model_a = joblib.load("processed/model-gauss.jbl")
    model_b = joblib.load("processed/model-multi.jbl")

    for model in [model_a, model_b]:
        pred_orig = model.predict(X_orig)[0]
        pred_mut = model.predict(X_mut)[0]

        assert pred_orig == pred_mut, (
            f"Inconsistent prediction for synonyms:\n"
            f"  '{original}' → {pred_orig}\n"
            f"  '{mutated}' → {pred_mut}"
        )

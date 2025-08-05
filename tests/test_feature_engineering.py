import pandas as pd
import pytest
import numpy as np
from app.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_data():
    """Provides sample student and reference data for testing."""
    student_df = pd.DataFrame({
        'id_soal': [1],
        'answer_text': ["teknologi informasi adalah studi tentang komputer"]
    })
    reference_df = pd.DataFrame({
        'id_soal': [1],
        'jawaban_referensi_terbaik': ["teknologi informasi adalah ilmu yang mempelajari komputer dan telekomunikasi"]
    })
    return student_df, reference_df


def test_feature_engineer_transform(sample_data):
    """
    Tests the fit and transform methods of the FeatureEngineer class.

    It verifies that the output is a NumPy array with the correct shape and that
    the calculated features have plausible values.
    """
    student_df, reference_df = sample_data

    # Arrange: Initialize the feature engineer and fit it on the sample data
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(student_df, reference_df)

    # Act: Transform the student data
    features = feature_engineer.transform(student_df, reference_df)

    # Assert: Check the results
    assert isinstance(features, np.ndarray), "Output should be a NumPy array"
    assert features.shape == (1, 3), "Output array should have shape (1, 3)"

    # Check the values of the features
    max_similarity = features[0, 0]
    mean_similarity = features[0, 1]
    length_ratio = features[0, 2]

    assert 0.0 <= max_similarity <= 1.0, f"Max cosine similarity should be between 0 and 1, but was {max_similarity}"
    assert 0.0 <= mean_similarity <= 1.0, f"Mean cosine similarity should be between 0 and 1, but was {mean_similarity}"
    assert length_ratio > 0, f"Length ratio should be a positive value, but was {length_ratio}"
    assert max_similarity > 0.4, f"Expected a higher max similarity for the given sentences, but got {max_similarity}"

"""
This module contains the FeatureEngineer class, responsible for transforming
raw text data into numerical features suitable for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise
from stop_words import get_stop_words
from tqdm import tqdm


class FeatureEngineer:
    """
    A class to handle all feature engineering tasks.

    This class encapsulates the TF-IDF vectorizer and the logic for creating
    similarity-based features. It can be fitted on a training dataset and then
    used to transform new data, ensuring consistency between training and prediction.
    """

    def __init__(self):
        """Initializes the FeatureEngineer with a TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(stop_words=get_stop_words('indonesian'))

    def fit(self, training_df: pd.DataFrame, reference_df: pd.DataFrame):
        """
        Fits the internal TF-IDF vectorizer on the combined vocabulary of
        training and reference answers.

        Args:
            training_df (pd.DataFrame): The training data containing student answers.
            reference_df (pd.DataFrame): The reference answer data.
        """
        all_text = pd.concat([
            training_df['answer_text'],
            reference_df['jawaban_referensi_terbaik']
        ], ignore_index=True)
        self.vectorizer.fit(all_text)
        return self

    def transform(self, student_answers_df: pd.DataFrame, reference_df: pd.DataFrame) -> np.ndarray:
        """
        Transforms student answers into the feature matrix using the fitted vectorizer.

        Args:
            student_answers_df (pd.DataFrame): A DataFrame of student answers.
            reference_df (pd.DataFrame): The reference answer data.

        Returns:
            np.ndarray: A NumPy array of shape (n_samples, 3) containing the features.
        """
        features_list = []
        ref_answers_by_question = reference_df.groupby('id_soal')['jawaban_referensi_terbaik'].apply(list)

        # Wrap the loop with tqdm for a progress bar
        for _, row in tqdm(student_answers_df.iterrows(), total=student_answers_df.shape[0], desc="  Feature Engineering"):
            student_answer, question_id = row['answer_text'], row['id_soal']
            top_answers = ref_answers_by_question.get(question_id, [])

            if not top_answers:
                features_list.append([0.0, 0.0, 0.0])
                continue

            student_vec = self.vectorizer.transform([student_answer])
            ref_vecs = self.vectorizer.transform(top_answers)
            similarities = pairwise.cosine_similarity(student_vec, ref_vecs)[0]

            features_list.append([
                np.max(similarities),
                np.mean(similarities),
                len(student_answer.split()) / np.mean([len(ans.split()) for ans in top_answers]) if top_answers else 0
            ])

        return np.array(features_list)
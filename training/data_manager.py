"""
This module handles all data fetching and database interactions for the training pipeline.
"""

import os
import pandas as pd
from sqlalchemy import create_engine

from .utils import create_safe_group_name
from app.logger_config import setup_logger
logger = setup_logger()


class DataManager:
    """
    Manages database connections and data retrieval for model training.
    """

    def __init__(self):
        """
        Initializes the DataManager and sets up the database engine.
        """
        db_url = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
        self.engine = create_engine(db_url)
        self._load_base_queries()

    def _load_base_queries(self):
        """Loads base SQL queries from files."""
        queries_dir = os.path.join(os.path.dirname(__file__), '..', 'queries')
        with open(os.path.join(queries_dir, 'get_training_data.sql'), 'r') as f:
            self.training_data_query_base = f.read().strip().rstrip(';')
        with open(os.path.join(queries_dir, 'get_reference_answers.sql'), 'r') as f:
            self.reference_answers_query_base = f.read().strip().rstrip(';')
        with open(os.path.join(queries_dir, 'get_all_training_groups.sql'), 'r') as f:
            self.get_groups_query = f.read()

    def get_training_groups(self) -> list:
        """
        Fetches all unique subject/grade combinations that have training data.

        Returns:
            list: A list of tuples, where each tuple is (subject_name, grade_level).
        """
        groups_df = pd.read_sql(self.get_groups_query, self.engine)
        groups_df.columns = ['mata_pelajaran', 'tingkat_kelas']
        return list(groups_df.itertuples(index=False, name=None))

    def get_training_data_for_group(self, subject_name: str, grade_level: str) -> pd.DataFrame:
        """
        Fetches the training data for a specific subject and grade level.

        Args:
            subject_name (str): The subject name.
            grade_level (str): The grade level.

        Returns:
            pd.DataFrame: A DataFrame containing the training data.
        """
        query = f"{self.training_data_query_base} AND mp.nama = %s AND tk.nama = %s"
        return pd.read_sql(query, self.engine, params=(subject_name, grade_level))

    def get_reference_answers_for_group(self, subject_name: str, grade_level: str) -> pd.DataFrame:
        """
        Fetches the reference answers for a specific subject and grade level.

        Args:
            subject_name (str): The subject name.
            grade_level (str): The grade level.

        Returns:
            pd.DataFrame: A DataFrame containing the reference answers.
        """
        query = f"{self.reference_answers_query_base} AND mp.nama = %s AND tk.nama = %s"
        return pd.read_sql(query, self.engine, params=(subject_name, grade_level))

    def export_data_to_excel(self, training_df: pd.DataFrame, reference_df: pd.DataFrame, subject_name: str, grade_level: str):
        """
        Exports the training and reference data for a group to a single Excel file.

        Args:
            training_df (pd.DataFrame): The training data.
            reference_df (pd.DataFrame): The reference answer data.
            subject_name (str): The subject name for creating the filename.
            grade_level (str): The grade level for creating the filename.
        """
        export_dir = os.path.join(os.path.dirname(__file__), '..', 'data_exports')
        os.makedirs(export_dir, exist_ok=True)

        safe_group_name = create_safe_group_name(subject_name, grade_level)
        filename = f"export_{safe_group_name}.xlsx"
        filepath = os.path.join(export_dir, filename)

        logger.info(f"Exporting data to Excel file: {filepath}")
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            training_df.to_excel(writer, sheet_name='Training Data', index=False)
            reference_df.to_excel(writer, sheet_name='Reference Answers', index=False)

    def close(self):
        """Closes the database connection pool."""
        if self.engine:
            self.engine.dispose()
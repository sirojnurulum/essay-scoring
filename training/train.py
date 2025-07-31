"""
Main script for training essay scoring models.

This script connects to the database, identifies all unique combinations of
subjects and grade levels that have training data, and then trains a separate
machine learning model for each combination.

The script can be run in several modes:
- Default (`make train`): Trains models for all groups that do not yet have a saved model file.
- Single Mode (`make train-next`): Trains only the next available model and then exits.
- Force Update (`make update-models`): Retrains all models, overwriting any existing ones.

The core logic involves feature engineering based on cosine similarity between
a student's answer and a set of reference answers, followed by training a
Linear Regression model. The final trained object is a scikit-learn Pipeline
containing both the TF-IDF vectorizer and the regression model.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from stop_words import get_stop_words
from sqlalchemy import create_engine
import sys
import os
import mysql.connector
from dotenv import load_dotenv

# Load environment variables (e.g., database credentials) from the .env file
load_dotenv()

print("Starting model training...")

def train_and_save_model_for_group(training_df, reference_df, subject_name, grade_level):
    """
    Trains, and saves a model for a specific subject and grade level combination.

    This function orchestrates the entire process for a single group:
    1.  Initializes and fits a TF-IDF vectorizer on both training and reference text.
    2.  Engineers features for each student answer based on its similarity to reference answers.
    3.  Trains a Linear Regression model on these features.
    4.  Bundles the vectorizer and model into a scikit-learn Pipeline.
    5.  Saves the pipeline to a .joblib file in the `app/models/` directory.

    Args:
        training_df (pd.DataFrame): DataFrame containing student answers and their true scores.
        reference_df (pd.DataFrame): DataFrame containing high-quality reference answers.
        subject_name (str): The name of the subject (e.g., "Biologi").
        grade_level (str): The grade level (e.g., "Kelas XII").
    """
    print("-" * 50)
    print(f"Training model for: {subject_name} - {grade_level} ({len(training_df)} records)")
    print(f"Using {len(reference_df)} top answers as reference.")

    # --- 2. Feature Engineering with Reference Answers ---
    # Use a standard list of Indonesian stop words to improve TF-IDF quality.
    stop_words_indonesian = get_stop_words('indonesian')
    vectorizer = TfidfVectorizer(stop_words=stop_words_indonesian)

    # Fit the vectorizer on a comprehensive vocabulary built from both student
    # answers and reference answers. This ensures all words are recognized.
    all_text = pd.concat([
        training_df['answer_text'],
        reference_df['jawaban_referensi_terbaik']
    ], ignore_index=True)
    vectorizer.fit(all_text)

    def create_features_with_references(training_df, reference_df, vectorizer):
        """
        Generates a feature matrix by comparing each student answer to its
        corresponding set of reference answers.

        For each student answer, it calculates three features:
        1.  max_sim: The highest cosine similarity to any reference answer.
        2.  avg_sim: The average cosine similarity across all reference answers.
        3.  len_ratio: The ratio of the student's answer length to the average
            length of the reference answers.

        Args:
            training_df (pd.DataFrame): The training data.
            reference_df (pd.DataFrame): The reference answer data.
            vectorizer (TfidfVectorizer): The pre-fitted TF-IDF vectorizer.

        Returns:
            np.ndarray: A NumPy array where each row is a feature vector for a student answer.
        """
        features_list = []
        # Group reference answers by question for quick lookup
        ref_answers_by_question = reference_df.groupby('id_soal')['jawaban_referensi_terbaik'].apply(list)

        for _, row in training_df.iterrows():
            student_answer = row['answer_text']
            question_id = row['id_soal']
            top_answers = ref_answers_by_question.get(question_id, [])

            if not top_answers:
                # Fallback: If no reference answers are found for a specific question,
                # create a zero-vector. This might happen with inconsistent data.
                features_list.append([0.0, 0.0, 0.0])
                continue

            # Vectorize student answer and reference answers
            student_vec = vectorizer.transform([student_answer])
            ref_vecs = vectorizer.transform(top_answers)

            # Calculate similarity scores
            similarities = cosine_similarity(student_vec, ref_vecs)[0]

            # Feature 1: Max similarity to a reference answer
            max_sim = np.max(similarities)
            # Feature 2: Average similarity to reference answers
            avg_sim = np.mean(similarities)
            # Feature 3: Length ratio compared to average length of reference answers
            avg_ref_len = np.mean([len(ans.split()) for ans in top_answers])
            len_ratio = len(student_answer.split()) / avg_ref_len if avg_ref_len > 0 else 0

            features_list.append([max_sim, avg_sim, len_ratio])
        
        return np.array(features_list)

    # Generate the feature matrix (X) and target vector (y) for training.
    X_train = create_features_with_references(training_df, reference_df, vectorizer)
    y_train = training_df['truth_score']

    # --- 3. Model Training ---
    # A simple Linear Regression model is used to map the features to a score.
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- 4. Model Serialization ---
    # The vectorizer and model are bundled into a single Pipeline object.
    # This makes prediction in the API much simpler, as it's a single object to call.
    pipeline = Pipeline([('vectorizer', vectorizer), ('model', model)])

    # Sanitize subject and grade names to create a robust, filesystem-friendly filename.
    safe_subject_name = subject_name.replace(' ', '_').lower()
    safe_grade_level = grade_level.replace(' ', '_').lower()
    # Use a clear separator '--' to distinguish subject from grade level.
    output_filename = f'model_{safe_subject_name}--{safe_grade_level}.joblib'
    # Ensure the target directory exists before saving.
    output_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'models', output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pipeline, output_path)

    print(f"Training complete for {subject_name} - {grade_level}. Model saved to: {output_path}")

def run_training_pipeline(force_update=False, single_mode=False):
    """
    Orchestrates the main training process.

    This function connects to the database, fetches all unique subject/grade
    groups, and then iterates through them. For each group, it fetches the
    necessary data and calls `train_and_save_model_for_group` unless a model
    already exists or `force_update` is specified.

    Args:
        force_update (bool): If True, all models will be retrained, overwriting existing files.
        single_mode (bool): If True, the script will train only one model and then exit.
    """

    engine = None
    try:
        print("Connecting to the database...")
        # Create a SQLAlchemy engine. This is the preferred way for pandas.
        db_url = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
        engine = create_engine(db_url)


        # 1. Get all unique groups (subject, grade) that have data
        print("Fetching all subject/grade groups that have training data...")
        get_groups_query = """
            SELECT DISTINCT mp.nama, tk.nama
            FROM accommodate_exam_student_answers sa
            JOIN question_banks qb ON sa.question_bank_id = qb.id
            JOIN mst_mata_pelajaran mp ON qb.subject_id = mp.id
            JOIN mst_tingkat_kelas tk ON qb.grade_level_id = tk.id
            WHERE sa.essay_answer IS NOT NULL;
        """
        groups_df = pd.read_sql(get_groups_query, engine)
        groups_df.columns = ['mata_pelajaran', 'tingkat_kelas']
        training_groups = list(groups_df.itertuples(index=False, name=None))
        
        if not training_groups:
            print("No training groups found in the database. Exiting.")
            return
        
        print(f"Found {len(training_groups)} unique groups to train.")

        # 2. Load the base SQL query for fetching training data
        # Reading from .sql files keeps the Python code cleaner.
        queries_dir = os.path.join(os.path.dirname(__file__), '..', 'queries')
        
        with open(os.path.join(queries_dir, 'get_training_data.sql'), 'r') as f:
            training_data_query_base = f.read().strip().rstrip(';')
            
        with open(os.path.join(queries_dir, 'get_reference_answers.sql'), 'r') as f:
            reference_answers_query_base = f.read().strip().rstrip(';')

        # 3. Loop through each group, fetch its specific data, and train
        for subject_name, grade_level in training_groups:
            # Construct the expected model filename to check for its existence.
            safe_subject_name_check = subject_name.replace(' ', '_').lower()
            safe_grade_level_check = grade_level.replace(' ', '_').lower()
            model_filename_check = f'model_{safe_subject_name_check}--{safe_grade_level_check}.joblib'
            model_path_check = os.path.join(os.path.dirname(__file__), '..', 'app', 'models', model_filename_check)

            model_exists = os.path.exists(model_path_check)

            # The core skipping logic: if the model exists and we are not forcing an
            # update, move to the next group in the loop.
            if not force_update and model_exists:
                print(f"Model for '{subject_name} - {grade_level}' already exists. Skipping...")
                continue

            # If we reach here, it means we need to train this model
            # (either it doesn't exist, or force_update is True).
            print("=" * 60)
            print(f"Processing group: {subject_name} - {grade_level}")

            # In single_mode, we announce that we've found the next target.
            if single_mode:
                print("Found next model to train...")

            # Fetch training data for the group
            group_training_query = f"{training_data_query_base} AND mp.nama = %s AND tk.nama = %s"
            training_df = pd.read_sql(group_training_query, engine, params=[subject_name, grade_level])

            # A sanity check: ensure there's a minimum amount of data to train a meaningful model.
            # This threshold can be adjusted.
            if len(training_df) < 20: # Example threshold
                print(f"Skipping group '{subject_name} - {grade_level}' due to insufficient training data ({len(training_df)} records).")
                continue
            
            print(f"Fetched {len(training_df)} training records for this group.")

            # Fetch reference answers for the group
            group_reference_query = f"{reference_answers_query_base} AND mp.nama = %s AND tk.nama = %s"
            reference_df = pd.read_sql(group_reference_query, engine, params=[subject_name, grade_level])

            if reference_df.empty:
                print(f"Warning: No reference answers found for '{subject_name} - {grade_level}'. Skipping training for this group.")
                continue

            # Train the model using both datasets
            train_and_save_model_for_group(training_df, reference_df, subject_name, grade_level)

            # If in single_mode, we exit the function immediately after one successful training.
            if single_mode:
                print("\nSingle training mode: Halting after one successful training.")
                return # Exit the function immediately.

        # This message is shown only if the loop completes.
        # If in single mode, reaching here means no new models were found to train.
        if single_mode:
            print("\nAll models appear to be trained. No new models to process.")

    except Exception as err:
        print(f"An error occurred during the training pipeline: {err}")
    finally:
        if engine:
            engine.dispose() # Closes all connections in the connection pool
            print("\nDatabase connection pool disposed. Training process finished.")

if __name__ == "__main__":
    # This block allows the script to be run from the command line with flags.
    # It checks for '--force-update' and '--train-next'.
    force_update_flag = '--force-update' in sys.argv
    single_mode_flag = '--train-next' in sys.argv

    # Prevent conflicting flags from being used together.
    if force_update_flag and single_mode_flag:
        print("Error: '--force-update' and '--train-next' are conflicting and cannot be used together.")
        sys.exit(1)

    if force_update_flag:
        print(">>> Force update mode enabled. All existing models will be retrained.")
    elif single_mode_flag:
        print(">>> Single training mode enabled. Will train the next available model and exit.")

    run_training_pipeline(force_update=force_update_flag, single_mode=single_mode_flag)
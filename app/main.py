"""
FastAPI application for the Essay Scoring System.

This module defines the main API endpoints for the essay scoring service.
It handles the following key responsibilities:
- Loading machine learning models and initializing a database connection pool on startup.
- Providing a `/score` endpoint to receive student essays and return a predicted score.
- Defining Pydantic models for robust request validation and response serialization.
- Interacting with a MySQL database to fetch reference answers for scoring.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Annotated
import joblib
import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. App and Model Loading ---

app = FastAPI(
    title="Essay Scoring System API",
    description="An API to score essays based on a reference answer.",
    version="1.0.0"
)

# Load environment variables (e.g., database credentials) from the .env file
load_dotenv()

# Define the absolute path to the directory where models are stored
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
# Global dictionary to cache loaded models in memory. Key: (subject, grade_level)
models = {}
# Global SQLAlchemy engine for database connections. Initialized at startup.
engine = None


def _load_all_models():
    """
    Scans the models directory, loads all .joblib model files into memory,
    and populates the global `models` dictionary.

    The model filenames are expected to follow the format:
    'model_{subject}--{grade_level}.joblib' (e.g., 'model_biologi--kelas_xii.joblib').

    Raises:
        RuntimeError: If the models directory does not exist.
    """
    global models
    if not os.path.isdir(MODELS_DIR):
        raise RuntimeError(
            f"Models directory not found: {MODELS_DIR}. Please train models first."
        )

    for model_file in os.listdir(MODELS_DIR):
        if model_file.endswith(".joblib"):
            # Filename example: model_biologi--kelas_xii.joblib
            clean_name = model_file.replace('model_', '').replace('.joblib', '')
            parts = clean_name.split("--")

            if len(parts) == 2:
                subject = parts[0].replace('_', ' ')
                grade = parts[1].replace('_', ' ')
                # Create a standardized key for the dictionary
                model_key = (subject.lower(), grade.lower())
                model_path = os.path.join(MODELS_DIR, model_file)
                # Load the model pipeline and store it in the global dictionary
                models[model_key] = joblib.load(model_path)
                print(f"Loaded model for: '{subject.title()} - {grade.title()}'")
    if not models:
        print("Warning: No models were found in the models directory.")


@app.on_event("startup")
def startup_event():
    """
    FastAPI startup event handler.

    This function is executed once when the application starts. It orchestrates
    the loading of all ML models and the initialization of the database
    connection pool (engine). This ensures that all necessary resources are
    ready before the API starts accepting requests.
    """
    global engine
    # 1. Load all machine learning models into the `models` dictionary
    _load_all_models()

    # 2. Initialize a database connection pool using SQLAlchemy
    print("Initializing database connection pool...")
    try:
        # Construct the database URL from environment variables
        db_url = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
        # Create the engine. `pool_recycle` prevents connection timeouts from the DB server.
        # `pool_pre_ping` checks if a connection is still alive before using it.
        engine = create_engine(db_url, pool_recycle=3600, pool_pre_ping=True)
        # Test the connection immediately to ensure the credentials and server are valid.
        # This allows the application to "fail fast" if the DB is unavailable.
        with engine.connect() as connection:
            print("Database connection successful.")
    except Exception as e:
        print(f"FATAL: Could not connect to database during startup: {e}")
        engine = None  # Ensure engine is None if connection fails


@app.on_event("shutdown")
def shutdown_event():
    """
    FastAPI shutdown event handler.

    This function is executed once when the application is shutting down.
    It safely disposes of the database connection pool, closing all
    connections and releasing resources.
    """
    if engine:
        engine.dispose()
        print("Database connection pool disposed.")


# --- 2. Pydantic Models for Request and Response ---


class EssayRequest(BaseModel):
    """
    Defines the structure and validation rules for the incoming request body
    to the /score endpoint.
    """
    subject: Annotated[
        str, Field(default=..., description="Subject of the essay, e.g., 'biologi'")
    ]
    grade_level: Annotated[
        str, Field(default=..., description="Grade level, e.g., 'kelas xii'")
    ]
    question_id: Annotated[
        int, Field(default=..., description="The unique ID of the question bank.")
    ]
    answer_text: Annotated[
        str, Field(default=..., description="The student's answer text.")
    ]


class ScoreResponse(BaseModel):
    """
    Defines the structure and validation rules for the response body sent
    from the /score endpoint.
    """
    score: Annotated[
        float,
        Field(
            default=...,
            ge=0.0,  # Value must be Greater than or Equal to 0.0
            le=10.0, # Value must be Less than or Equal to 10.0
            description="The predicted score, clamped between 0.0 and 10.0",
            example=8.5,
        ),
    ]


# --- 3. API Endpoint ---


@app.post("/score", response_model=ScoreResponse)
def predict_score(request: EssayRequest):
    """
    Calculates and returns a score for a given student's essay answer.

    This endpoint performs the following steps:
    1.  Validates the incoming request against the `EssayRequest` model.
    2.  Selects the appropriate pre-trained model based on `subject` and `grade_level`.
    3.  Connects to the database to fetch the reference answers for the given `question_id`.
    4.  Performs feature engineering by comparing the student's answer to the reference answers.
    5.  Uses the model to predict a raw score based on the generated features.
    6.  Clamps the score to a 0-10 range and rounds it.
    7.  Returns the final score in a JSON response, validated against the `ScoreResponse` model.

    Args:
        request (EssayRequest): The request body containing essay details.

    Raises:
        HTTPException(503): If the models or database connection are not available (e.g., failed at startup).
        HTTPException(404): If no model is found for the requested subject and grade level.
        HTTPException(500): For any other unexpected errors during the scoring process.

    Returns:
        ScoreResponse: A dictionary containing the final calculated score.
    """
    # Fail early if the service is not ready
    if not models or not engine:
        raise HTTPException(
            status_code=503,
            detail="Service is not ready. Models or database connection are unavailable.",
        )

    # Create the key to look up the correct model from the `models` dictionary
    model_key = (request.subject.lower(), request.grade_level.lower())
    if model_key not in models:
        # If no model exists for the combination, return a helpful error message
        raise HTTPException(
            status_code=404,
            detail=f"No model found for '{request.subject} - {request.grade_level}'. Available models: {[f'{s.title()} - {g.title()}' for s, g in models.keys()]}",
        )

    try:
        # Retrieve the correct pipeline (vectorizer + model) from the cache
        model_pipeline = models[model_key]

        # Load the base SQL query from the external .sql file
        queries_dir = os.path.join(os.path.dirname(__file__), '..', 'queries')
        with open(os.path.join(queries_dir, 'get_reference_answers.sql'), 'r') as f:
            reference_answers_query_base = f.read().strip().rstrip(";")

        # Fetch reference answers for the specific question using the global engine
        group_reference_query = f"{reference_answers_query_base} AND qb.id = %s"
        # `pd.read_sql` uses the connection pool from the engine to execute the query
        reference_df = pd.read_sql(
            group_reference_query, engine, params=[request.question_id]
        )

        # Extract the two main components from the loaded pipeline
        vectorizer = model_pipeline.named_steps['vectorizer']
        model = model_pipeline.named_steps['model']

        # --- Replicate Feature Engineering from Training ---
        # This logic must be identical to the feature creation in `training/train.py`
        top_answers = reference_df['jawaban_referensi_terbaik'].tolist()

        if not top_answers:
            # Fallback: If no reference answers are found for this specific question,
            # we cannot calculate similarity. We create zero-value features,
            # which will likely result in a low or zero score from the model.
            features = np.array([[0.0, 0.0, 0.0]])
        else:
            # Transform the student's answer and reference answers into TF-IDF vectors
            student_vec = vectorizer.transform([request.answer_text])
            ref_vecs = vectorizer.transform(top_answers)

            # Calculate cosine similarity between the student's answer and each reference answer
            similarities = cosine_similarity(student_vec, ref_vecs)[0]

            # Feature 1: Maximum similarity (how close is it to the best matching answer?)
            max_sim = np.max(similarities)
            # Feature 2: Average similarity (how close is it to all answers on average?)
            avg_sim = np.mean(similarities)
            # Feature 3: Length ratio (is the answer length comparable to the references?)
            avg_ref_len = np.mean([len(ans.split()) for ans in top_answers])
            len_ratio = (
                len(request.answer_text.split()) / avg_ref_len if avg_ref_len > 0 else 0
            )

            # Combine features into a single array for the model
            features = np.array([[max_sim, avg_sim, len_ratio]])

        # --- Prediction ---
        # Use the trained model to predict a score from the features
        predicted_score = model.predict(features)[0]

        # Post-processing: Clamp the score to a valid 0-10 range to handle outliers
        final_score = max(0.0, min(predicted_score, 10.0))

        # Return a dictionary. FastAPI will automatically validate it against ScoreResponse.
        return {"score": round(final_score, 2)}
    except Exception as e:
        # A general catch-all for any unexpected errors (e.g., database errors,
        # malformed data, etc.) to prevent the server from crashing.
        raise HTTPException(
            status_code=500, detail=f"An error occurred during prediction: {str(e)}"
        )

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
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from .feature_engineering import FeatureEngineer
from .logger_config import setup_logger
# --- 1. App and Model Loading ---

app = FastAPI(
    title="Essay Scoring System API",
    description="An API to score essays based on a reference answer.",
    version="1.0.0"
)

logger = setup_logger()

# Load environment variables (e.g., database credentials) from the .env file
load_dotenv()

# Determine which model type the API should serve, default to 'xgboost'
MODEL_TYPE = os.getenv('MODEL_TYPE', 'xgboost').lower()

# Define the absolute path to the directory where models are stored
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models', MODEL_TYPE)
# Global dictionary to cache loaded models in memory. Key: (subject, grade_level)
models = {}
# Global SQLAlchemy engine for database connections. Initialized at startup.
engine = None


def _load_all_models():
    """
    Scans the appropriate models directory based on MODEL_TYPE, loads all
    model assets into memory, and populates the global `models` dictionary.

    Raises:
        RuntimeError: If the models directory does not exist.
    """
    global models
    logger.info(f"--- Loading models of type: {MODEL_TYPE.upper()} ---")
    if not os.path.isdir(MODELS_DIR):
        raise RuntimeError(
            f"Models directory not found: {MODELS_DIR}. Please train '{MODEL_TYPE}' models first."
        )

    for model_dir_name in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_dir_name)
        if os.path.isdir(model_path):
            try:
                # Directory name format: {subject}--{grade_level}
                parts = model_dir_name.split("--")
                if len(parts) != 2: continue

                subject = parts[0].replace('_', ' ')
                grade = parts[1].replace('_', ' ')
                # Create a standardized key for the dictionary
                model_key = (subject.lower(), grade.lower())

                if MODEL_TYPE == 'xgboost':
                    # Load the entire scikit-learn pipeline
                    feature_engineer = joblib.load(os.path.join(model_path, 'feature_engineer.joblib'))
                    model = joblib.load(os.path.join(model_path, 'model.joblib'))
                    models[model_key] = {'feature_engineer': feature_engineer, 'model': model}
                elif MODEL_TYPE == 'deep-learning':
                    # Load vectorizer and Keras model separately
                    feature_engineer = joblib.load(os.path.join(model_path, 'feature_engineer.joblib'))
                    model = tf.keras.models.load_model(os.path.join(model_path, 'model.keras'))
                    models[model_key] = {'feature_engineer': feature_engineer, 'model': model}
                
                logger.info(f"Loaded model for: '{subject.title()} - {grade.title()}'")
            except Exception as e:
                logger.warning(f"Could not load model from directory '{model_dir_name}'. Reason: {e}")
    if not models:
        logger.warning("No models were found in the models directory.")


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
    logger.info("Initializing database connection pool...")
    try:
        # Construct the database URL from environment variables
        db_url = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
        # Create the engine. `pool_recycle` prevents connection timeouts from the DB server.
        # `pool_pre_ping` checks if a connection is still alive before using it.
        engine = create_engine(db_url, pool_recycle=3600, pool_pre_ping=True)
        # Test the connection immediately to ensure the credentials and server are valid.
        # This allows the application to "fail fast" if the DB is unavailable.
        with engine.connect() as connection:
            logger.info("Database connection successful.")
    except Exception as e:
        logger.critical(f"Could not connect to database during startup: {e}")
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
        logger.info("Database connection pool disposed.")


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
    logger.info(
        f"Received scoring request for QID:{request.question_id} "
        f"({request.subject.title()} - {request.grade_level.title()})"
    )
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
        # Retrieve the cached model assets
        # The asset structure is now consistent for both model types
        model_assets = models[model_key]
        feature_engineer = model_assets['feature_engineer']
        model = model_assets['model']

        # Load the base SQL query from the external .sql file
        queries_dir = os.path.join(os.path.dirname(__file__), '..', 'queries')
        with open(os.path.join(queries_dir, 'get_reference_answers.sql'), 'r') as f:
            reference_answers_query_base = f.read().strip().rstrip(";")

        # Fetch reference answers for the specific question using the global engine
        group_reference_query = f"{reference_answers_query_base} AND qb.id = %s"
        # `pd.read_sql` uses the connection pool from the engine to execute the query
        reference_df = pd.read_sql(
            group_reference_query, engine, params=(request.question_id,)
        )

        # --- Feature Engineering using the dedicated class ---
        # Create a DataFrame for the single student answer to match the expected input format.
        student_df = pd.DataFrame([{'answer_text': request.answer_text, 'id_soal': request.question_id}])
        # Use the feature engineer to create features.
        features = feature_engineer.transform(student_df, reference_df)

        # --- Prediction ---
        # Use the trained model to predict a score from the features
        raw_prediction = model.predict(features)
        # Keras/TF returns a 2D array (e.g., [[9.5]]), so we extract the scalar value.
        # For scikit-learn/XGBoost, it's a 1D array (e.g., [9.5]), so [0] is sufficient.
        predicted_score = raw_prediction[0][0] if isinstance(raw_prediction, np.ndarray) and raw_prediction.ndim > 1 else raw_prediction[0]


        # Post-processing: Clamp the score to a valid 0-10 range to handle outliers
        final_score = max(0.0, min(predicted_score, 10.0))

        # Return a dictionary. FastAPI will automatically validate it against ScoreResponse.
        return {"score": round(final_score, 2)}
    except Exception as e:
        # A general catch-all for any unexpected errors (e.g., database errors,
        # malformed data, etc.) to prevent the server from crashing.
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred during prediction."
        )

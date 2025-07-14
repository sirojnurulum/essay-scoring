from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. App and Model Loading ---

app = FastAPI(
    title="Essay Scoring System API",
    description="An API to score essays based on a reference answer.",
    version="1.0.0"
)

# Define the path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.joblib')
model_pipeline = None

@app.on_event("startup")
def load_model():
    """Load the model when the application starts."""
    global model_pipeline
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model file not found. Please train the model first by running training/train.py")
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")

# --- 2. Pydantic Models for Request and Response ---

class EssayRequest(BaseModel):
    subject: str = Field(..., example="Biology")
    essay_text: str = Field(..., example="The mitochondria is a part of the cell.")
    answer_text: str = Field(..., example="The mitochondrion is an organelle found in large numbers in most cells, in which the biochemical processes of respiration and energy production occur.")

class ScoreResponse(BaseModel):
    score: float = Field(..., example=8.5)

# --- 3. API Endpoint ---

@app.post("/score", response_model=ScoreResponse)
def predict_score(request: EssayRequest):
    """
    Receives essay and answer text, and returns a predicted score.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or available.")

    try:
        # Extract vectorizer and model from the pipeline
        vectorizer = model_pipeline.named_steps['vectorizer']
        model = model_pipeline.named_steps['model']

        # --- Feature Engineering (must be identical to training) ---
        essay_vec = vectorizer.transform([request.essay_text])
        answer_vec = vectorizer.transform([request.answer_text])

        sim_score = cosine_similarity(essay_vec, answer_vec)[0][0]
        len_ratio = len(request.essay_text.split()) / len(request.answer_text.split()) if len(request.answer_text.split()) > 0 else 0

        features = np.array([[sim_score, len_ratio]])

        # --- Prediction ---
        predicted_score = model.predict(features)[0]
        
        # Clamp the score to a reasonable range, e.g., 0-10
        final_score = max(0.0, min(predicted_score, 10.0))

        return ScoreResponse(score=round(final_score, 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

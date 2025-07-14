import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

print("Starting model training...")

# --- 1. Data Simulation ---
# In a real-world scenario, you would load your data from a database or CSV file.
# The data should contain 'essay_text', 'answer_text', and the ground 'truth_score'.
# For demonstration, we create a dummy dataset.

data = {
    'essay_text': [
        "The sun is a star. It is very hot.",
        "The sun is the center of our solar system. It provides light and heat.",
        "Photosynthesis is how plants make food.",
        "The mitochondria is the powerhouse of the cell.",
        "The mitochondria is a part of the cell that generates energy."
    ],
    'answer_text': [
        "The sun is a star at the center of the Solar System. It is a nearly perfect ball of hot plasma.",
        "The sun is a star at the center of the Solar System. It is a nearly perfect ball of hot plasma.",
        "The process by which green plants use sunlight to synthesize foods from carbon dioxide and water.",
        "The mitochondrion is an organelle found in large numbers in most cells, in which the biochemical processes of respiration and energy production occur.",
        "The mitochondrion is an organelle found in large numbers in most cells, in which the biochemical processes of respiration and energy production occur."
    ],
    # Scores are typically between 0 and 10, or 0 and 100. Let's use 0-10.
    'truth_score': [7.5, 9.0, 4.0, 8.0, 9.5]
}
df = pd.DataFrame(data)

print(f"Loaded {len(df)} sample records for training.")

# --- 2. Feature Engineering ---
# We create features that the model can learn from.

print("Performing feature engineering...")

# Vectorizer to convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english')

# We need to fit the vectorizer on both essays and answers to have a shared vocabulary
all_text = pd.concat([df['essay_text'], df['answer_text']], ignore_index=True)
vectorizer.fit(all_text)

def create_features(essays, answers):
    """Creates feature vectors from essay and answer texts."""
    essay_vectors = vectorizer.transform(essays)
    answer_vectors = vectorizer.transform(answers)
    
    # Feature 1: Cosine Similarity
    sim_scores = np.array([cosine_similarity(ev.reshape(1, -1), av.reshape(1, -1))[0][0] for ev, av in zip(essay_vectors, answer_vectors)])
    
    # Feature 2: Length Ratio
    len_ratio = np.array([len(e.split()) / len(a.split()) if len(a.split()) > 0 else 0 for e, a in zip(essays, answers)])
    
    return np.vstack([sim_scores, len_ratio]).T

X_train = create_features(df['essay_text'], df['answer_text'])
y_train = df['truth_score']

# --- 3. Model Training ---
print("Training the regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# --- 4. Model Serialization ---
# We save the vectorizer and the trained model together in a single file.
pipeline = Pipeline([('vectorizer', vectorizer), ('model', model)])

output_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'models', 'model.joblib')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
joblib.dump(pipeline, output_path)

print(f"Training complete. Model saved to: {output_path}")
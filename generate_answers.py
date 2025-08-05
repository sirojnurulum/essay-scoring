"""
Script to generate high-quality reference answers for questions in the database.

This script connects to the database, fetches all questions from the question_banks
table, uses the Gemini Pro model to generate a detailed answer for each, and
exports the results to an Excel file for manual review and potential use as
new reference data.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from tqdm import tqdm
import google.generativeai as genai

from app.logger_config import setup_logger

logger = setup_logger()

def main():
    """Main function to orchestrate the answer generation process."""
    load_dotenv()

    # Configure the Gemini API
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in .env file. Please add it.")
        return
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')

    # Connect to the database
    db_url = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    engine = create_engine(db_url)

    try:
        logger.info("Fetching questions from the database...")
        with open('queries/get_all_questions.sql', 'r') as f:
            query = f.read()
        questions_df = pd.read_sql(query, engine)
        logger.info(f"Found {len(questions_df)} questions to answer.")

        results = []
        prompt_template = (
            "Anda adalah seorang ahli yang memberikan jawaban referensi berkualitas tinggi untuk soal esai. "
            "Jawablah pertanyaan berikut dengan jelas, akurat, dan lengkap. "
            "Pertanyaan: {question_text}"
        )

        for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Generating Answers"):
            question_id = row['id']
            question_text = row['question']

            try:
                prompt = prompt_template.format(question_text=question_text)
                response = model.generate_content(prompt)
                generated_answer = response.text
            except Exception as e:
                logger.warning(f"Could not generate answer for QID {question_id}. Reason: {e}")
                generated_answer = "ERROR: Could not generate answer."

            results.append({
                'question_id': question_id,
                'question_text': question_text,
                'generated_answer': generated_answer
            })

        # Export results to Excel
        output_dir = 'generated_data'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'generated_reference_answers.xlsx')
        
        logger.info(f"Exporting {len(results)} generated answers to {output_path}...")
        results_df = pd.DataFrame(results)
        results_df.to_excel(output_path, index=False, engine='openpyxl')
        logger.info("Export complete.")

    finally:
        engine.dispose()
        logger.info("Database connection closed.")

if __name__ == "__main__":
    main()
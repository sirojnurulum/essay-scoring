# Makefile for the Essay Scoring System.
#
# This file provides a set of commands to automate common development tasks such as
# setting up the environment, running the training scripts for different model
# types, and managing the application server and Docker containers.

# Define variables for Python interpreter and virtual environment directory
PYTHON := python3
VENV_DIR := .venv

# Set the default command to run when 'make' is called without arguments
.DEFAULT_GOAL := help

# Phony targets are not files, they are commands
.PHONY: help setup generate-answers train train-next update-models train-dl train-next-dl update-models-dl run clean docker-build docker-up docker-down docker-logs

help:
	@echo "Available commands:"
	@echo "  setup          - Install dependencies using Poetry"
	@echo ""
	@echo "--- Data Generation ---"
	@echo "  generate-answers - Use AI to generate new reference answers for all questions"
	@echo ""
	@echo "--- XGBoost Model Training ---"
	@echo "  train          - Train XGBoost models (skips existing models)"
	@echo "  train-next     - Train the next available XGBoost model"
	@echo "  update-models  - Force re-training of all XGBoost models"
	@echo ""
	@echo "--- Deep Learning Model Training ---"
	@echo "  train-dl       - Train Deep Learning models (skips existing)"
	@echo "  train-next-dl  - Train the next single Deep Learning model"
	@echo "  update-models-dl - Force re-training of all Deep Learning models"
	@echo ""
	@echo "--- Quality & Testing ---"
	@echo "  test           - Run all automated tests with pytest"
	@echo ""
	@echo "  run            - Run the FastAPI server locally for development"
	@echo "  clean          - Remove virtual environment and cache files"
	@echo "  docker-build   - Build the Docker image for the application"
	@echo "  docker-up      - Run the application and its services using Docker Compose"
	@echo "  docker-down    - Stop and remove the Docker containers"
	@echo "  docker-logs    - View logs from the running Docker containers"

setup:
	@echo ">>> Installing dependencies with Poetry..."
	@poetry install
	@echo ">>> Dependencies installed."

generate-answers:
	@echo ">>> Generating new reference answers using Gemini AI..."
	@poetry run $(PYTHON) generate_answers.py

train:
	@echo ">>> Running XGBoost model training script (skipping existing models)..."
	@poetry run $(PYTHON) -m training.train --model-type xgboost

train-next:
	@echo ">>> Training the next available XGBoost model..."
	@poetry run $(PYTHON) -m training.train --model-type xgboost --train-next

update-models:
	@echo ">>> Forcing re-training of all XGBoost models..."
	@poetry run $(PYTHON) -m training.train --model-type xgboost --force-update

train-dl:
	@echo ">>> Running Deep Learning model training script (skipping existing models)..."
	@poetry run $(PYTHON) -m training.train --model-type deep-learning

train-next-dl:
	@echo ">>> Training the next available Deep Learning model..."
	@poetry run $(PYTHON) -m training.train --model-type deep-learning --train-next

update-models-dl:
	@echo ">>> Forcing re-training of all Deep Learning models..."
	@poetry run $(PYTHON) -m training.train --model-type deep-learning --force-update

test:
	@echo ">>> Running all automated tests..."
	@poetry run pytest

run:
	@echo ">>> Starting FastAPI server locally on http://127.0.0.1:8000"
	@poetry run uvicorn app.main:app --reload

clean:
	@echo ">>> Removing virtual environment and cache files..."
	@rm -rf $(VENV_DIR)
	@rm -rf .pytest_cache
	@find . -type d -name "__pycache__" -exec rm -r {} +
	@echo ">>> Cleanup complete."

docker-build:
	@echo ">>> Building Docker image..."
	@docker-compose build

docker-up:
	@echo ">>> Starting application with Docker Compose..."
	@docker-compose up -d

docker-down:
	@echo ">>> Stopping and removing Docker containers..."
	@docker-compose down

docker-logs:
	@echo ">>> Tailing logs from Docker container..."
	@docker-compose logs -f

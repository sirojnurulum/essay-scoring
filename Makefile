# Use bash for more advanced features
SHELL := /bin/bash
# Define the virtual environment directory
VENV_DIR := venv
PYTHON := $(VENV_DIR)/bin/python3
PIP := $(VENV_DIR)/bin/pip3

# Default command when running `make`
.DEFAULT_GOAL := help

# Phony targets are not files, they are commands
.PHONY: help setup install train run clean docker-build docker-up docker-down docker-logs

help:
	@echo "Available commands:"
	@echo "  setup          - Create virtual environment and install dependencies"
	@echo "  install        - Install/update dependencies from requirements.txt"
	@echo "  train          - Run the model training script"
	@echo "  run            - Run the FastAPI server locally for development"
	@echo "  clean          - Remove virtual environment and cache files"
	@echo "  docker-build   - Build the Docker image for the application"
	@echo "  docker-up      - Start the application using Docker Compose"
	@echo "  docker-down    - Stop and remove the application containers"
	@echo "  docker-logs    - View logs from the running container"

setup: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements.txt
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo ">>> Creating virtual environment..."; \
		python3 -m venv $(VENV_DIR); \
	fi
	@echo ">>> Activating virtual environment and installing dependencies..."
	@. $(VENV_DIR)/bin/activate; $(PIP) install -r requirements.txt
	@touch $(VENV_DIR)/bin/activate

install:
	@. $(VENV_DIR)/bin/activate; $(PIP) install -r requirements.txt
	@echo ">>> Dependencies installed."

train:
	@echo ">>> Running model training script..."
	@. $(VENV_DIR)/bin/activate; $(PYTHON) training/train.py

run:
	@echo ">>> Starting FastAPI server locally on http://127.0.0.1:8000"
	@. $(VENV_DIR)/bin/activate; uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

clean:
	@echo ">>> Cleaning up generated files..."
	@rm -rf $(VENV_DIR) __pycache__ */__pycache__ .pytest_cache .coverage

docker-build:
	@echo ">>> Building Docker image..."
	@docker build -t essay-scorer-api .

docker-up:
	@echo ">>> Starting services with Docker Compose..."
	@docker-compose up --build -d

docker-down:
	@echo ">>> Stopping services with Docker Compose..."
	@docker-compose down

docker-logs:
	@echo ">>> Tailing logs for scorer-api service..."
	@docker-compose logs -f scorer-api
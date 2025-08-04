"""
This module contains classes for training different types of models.

Each trainer class is responsible for the entire lifecycle of a specific model
type, including feature engineering, evaluation (e.g., cross-validation or
hyperparameter tuning), final model fitting, and saving the necessary artifacts.
"""

import os
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from xgboost import XGBRegressor

from app.feature_engineering import FeatureEngineer
from app.logger_config import setup_logger

logger = setup_logger()


class BaseModelTrainer:
    """Abstract base class for model trainers."""

    def __init__(self, subject_name: str, grade_level: str, model_type: str):
        self.subject_name = subject_name
        self.grade_level = grade_level
        self.model_type = model_type
        self.feature_engineer = FeatureEngineer()
        self.model = None

        # Define the output directory for model artifacts
        safe_subject = subject_name.replace(' ', '_').lower()
        safe_grade = grade_level.replace(' ', '_').lower()
        model_group_name = f'{safe_subject}--{safe_grade}'
        self.model_dir = os.path.join(os.path.dirname(__file__), '..', 'app', 'models', model_type, model_group_name)
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, training_df, reference_df):
        """Main training orchestration method."""
        logger.info("-" * 50)
        logger.info(f"Begin training for: {self.subject_name} - {self.grade_level}")

        logger.info("Fitting FeatureEngineer (TF-IDF vectorizer)...")
        self.feature_engineer.fit(training_df, reference_df)
        logger.info(f"Vectorizer fit complete. Vocabulary size: {len(self.feature_engineer.vectorizer.get_feature_names_out())} words.")

        logger.info("Transforming training data into features...")
        X_train = self.feature_engineer.transform(training_df, reference_df)
        y_train = training_df['truth_score']

        self._train_model(X_train, y_train)
        self._save_artifacts()

        logger.info(f"Training complete. Model assets saved to: {self.model_dir}")

    def _train_model(self, X_train, y_train):
        """Placeholder for model-specific training logic."""
        raise NotImplementedError("This method must be implemented by subclasses.")

    def _save_artifacts(self):
        """Placeholder for model-specific saving logic."""
        raise NotImplementedError("This method must be implemented by subclasses.")


class XGBoostTrainer(BaseModelTrainer):
    """Trainer for XGBoost models, including hyperparameter tuning."""

    def _train_model(self, X_train, y_train):
        logger.info("Performing GridSearchCV for XGBoost hyperparameter tuning...")
        param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
        xgb = XGBRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5, verbose=1)
        grid_search.fit(X_train, y_train)

        logger.info(f"GridSearchCV Complete. Best MAE: {-grid_search.best_score_:.4f}")
        logger.info(f"Best Parameters found: {grid_search.best_params_}")
        self.model = grid_search.best_estimator_

    def _save_artifacts(self):
        joblib.dump(self.feature_engineer, os.path.join(self.model_dir, 'feature_engineer.joblib'))
        joblib.dump(self.model, os.path.join(self.model_dir, 'model.joblib'))


class DeepLearningTrainer(BaseModelTrainer):
    """Trainer for Deep Learning models, including cross-validation."""

    def _build_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_absolute_error')
        return model

    def _evaluate(self, X_train, y_train):
        # EarlyStopping callback to prevent overfitting and find the best model.
        # It stops training if 'val_loss' doesn't improve for 5 epochs.
        # `restore_best_weights=True` ensures the model has the weights from the best epoch.
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        logger.info("Performing 5-fold cross-validation for Deep Learning model...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
            logger.info(f"  Processing CV fold {fold + 1}/5...")
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            
            eval_model = self._build_model(input_shape=(X_train_fold.shape[1],))
            # Use 10% of the fold's training data as a validation set for early stopping.
            eval_model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=32, verbose=0, validation_split=0.1, callbacks=[early_stopping_callback])
            preds = eval_model.predict(X_val_fold).flatten()
            cv_scores.append(mean_absolute_error(y_val_fold, preds))
        logger.info(f"Cross-Validation Complete. Mean Absolute Error: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    def _train_model(self, X_train, y_train):
        self._evaluate(X_train, y_train)
        logger.info("-" * 50)
        logger.info("Training final Deep Learning model on the entire dataset...")
        self.model = self._build_model(input_shape=(X_train.shape[1],))
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
        # Use 10% of the full training data for validation during final training.
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.1, callbacks=[early_stopping_callback])

    def _save_artifacts(self):
        joblib.dump(self.feature_engineer, os.path.join(self.model_dir, 'feature_engineer.joblib'))
        self.model.save(os.path.join(self.model_dir, 'model.keras'))
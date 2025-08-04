"""
Main script for training essay scoring models.

This script acts as an orchestrator for the training pipeline. It uses a
DataManager to fetch data and a ModelTrainer to handle the ML-specific tasks
like feature engineering, evaluation, and model saving.

The script can be run in several modes:
- Default (`make train`): Trains models for all groups that do not yet have a saved model file.
- Single Mode (`make train-next`): Trains only the next available model and then exits.
- Force Update (`make update-models`): Retrains all models, overwriting any existing ones.
"""
import sys
import os
from dotenv import load_dotenv

from app.logger_config import setup_logger
from .data_manager import DataManager
from .model_trainer import XGBoostTrainer, DeepLearningTrainer

load_dotenv()
logger = setup_logger()


def get_trainer(model_type, subject_name, grade_level):
    """Factory function to get the correct trainer instance."""
    if model_type == 'xgboost':
        return XGBoostTrainer(subject_name, grade_level, model_type)
    elif model_type == 'deep-learning':
        return DeepLearningTrainer(subject_name, grade_level, model_type)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_training_pipeline(model_type, force_update=False, single_mode=False):
    """
    Orchestrates the main training process.

    This function connects to the database, fetches all unique subject/grade
    groups, and then iterates through them, delegating the training of each
    group to the appropriate ModelTrainer class.

    Args:
        model_type (str): The type of model to train ('xgboost' or 'deep-learning').
        force_update (bool): If True, all models will be retrained, overwriting existing files.
        single_mode (bool): If True, the script will train only one model and then exit.
    """

    data_manager = None
    try:
        data_manager = DataManager()
        logger.info("Fetching all subject/grade groups that have training data...")
        training_groups = data_manager.get_training_groups()

        if not training_groups:
            logger.warning("No training groups found in the database. Exiting.")
            return
        logger.info(f"Found {len(training_groups)} unique groups to train.")

        # Loop through each group, fetch its specific data, and train
        for subject_name, grade_level in training_groups:
            # Construct the expected model filename to check for its existence.
            safe_subject_name_check = subject_name.replace(' ', '_').lower()
            safe_grade_level_check = grade_level.replace(' ', '_').lower()
            model_group_name_check = f'{safe_subject_name_check}--{safe_grade_level_check}'
            model_path_check = os.path.join(os.path.dirname(__file__), '..', 'app', 'models', model_type, model_group_name_check)

            model_exists = os.path.exists(model_path_check)

            # The core skipping logic: if the model exists and we are not forcing an
            # update, move to the next group in the loop.
            if not force_update and model_exists:
                logger.info(f"Model for '{subject_name} - {grade_level}' already exists. Skipping...")
                continue

            # If we reach here, it means we need to train this model
            # (either it doesn't exist, or force_update is True).
            logger.info("=" * 60)
            logger.info(f"Processing group: {subject_name} - {grade_level}")

            # In single_mode, we announce that we've found the next target.
            if single_mode:
                logger.info("Found next model to train...")

            # Fetch training data for the group
            training_df = data_manager.get_training_data_for_group(subject_name, grade_level)

            # A sanity check: ensure there's a minimum amount of data to train a meaningful model.
            # This threshold can be adjusted.
            if len(training_df) < 20: # Example threshold
                logger.warning(f"Skipping group '{subject_name} - {grade_level}' due to insufficient training data ({len(training_df)} records).")
                continue
            
            logger.info(f"Fetched {len(training_df)} training records for this group.")

            # Fetch reference answers for the group
            logger.info("Fetching reference answers...")
            reference_df = data_manager.get_reference_answers_for_group(subject_name, grade_level)

            if reference_df.empty:
                logger.warning(f"No reference answers found for '{subject_name} - {grade_level}'. Skipping training for this group.")
                continue

            trainer = get_trainer(model_type, subject_name, grade_level)
            trainer.train(training_df, reference_df)

            # If in single_mode, we exit the function immediately after one successful training.
            if single_mode:
                logger.info("Single training mode: Halting after one successful training.")
                return # Exit the function immediately.

        # This message is shown only if the loop completes.
        # If in single mode, reaching here means no new models were found to train.
        if single_mode:
            logger.info("All models appear to be trained. No new models to process.")

    except Exception as err:
        logger.exception(f"An unhandled error occurred during the training pipeline: {err}")
    finally:
        if data_manager:
            data_manager.close()
            logger.info("Database connection pool disposed. Training process finished.")

if __name__ == "__main__":
    # This block allows the script to be run from the command line with flags.
    # It checks for '--force-update', '--train-next', and '--model-type'.
    force_update_flag = '--force-update' in sys.argv
    single_mode_flag = '--train-next' in sys.argv
    
    # Determine model type from arguments, default to 'xgboost'
    model_type_arg = 'xgboost'
    if '--model-type' in sys.argv:
        try:
            model_type_index = sys.argv.index('--model-type') + 1
            model_type_arg = sys.argv[model_type_index]
        except (ValueError, IndexError):
            logger.error("Error: --model-type flag requires an argument ('xgboost' or 'deep-learning').")
            sys.exit(1)

    # Prevent conflicting flags from being used together.
    if force_update_flag and single_mode_flag:
        logger.error("Error: '--force-update' and '--train-next' are conflicting and cannot be used together.")
        sys.exit(1)

    logger.info(f">>> Mode: {model_type_arg.upper()} | Force Update: {force_update_flag} | Single Mode: {single_mode_flag} <<<")

    run_training_pipeline(model_type=model_type_arg, force_update=force_update_flag, single_mode=single_mode_flag)
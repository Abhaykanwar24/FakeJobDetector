from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.logger import logging

def main():
    try:
        logging.info("===== Data Ingestion Started =====")
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info("===== Data Ingestion Completed =====\n")

        logging.info("===== Data Transformation Started =====")
        data_transformation = DataTransformation()
        train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path, test_path
        )
        logging.info("===== Data Transformation Completed =====\n")

        logging.info("===== Model Training Started =====")
        model_trainer = ModelTrainer()
        model_path, best_model_name, best_score = model_trainer.initiate_model_trainer(
            train_array, test_array, preprocessor_path
        )
        logging.info(f"===== Model Training Completed =====")
        logging.info(f"Best Model: {best_model_name} | F1-score: {best_score:.4f} | Saved at: {model_path}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()

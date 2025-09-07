import os
import sys
from dataclasses import dataclass

import xgboost as xgb


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts" , "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("Splitiing training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info(f"Original training set shape: {X_train.shape}, {y_train.shape}")

            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            logging.info(f"After SMOTE, training set shape: {X_train.shape}, {y_train.shape}")

            # Only XGBoost with GPU
            model_name = 'XGBoost'
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            )

            params = {
                "n_estimators": [100, 200, 300, 400],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "gamma": [0, 0.1, 0.2, 0.3],
                "reg_lambda": [1, 1.5, 2.0]
            }

            logging.info(f"Starting RandomizedSearchCV for {model_name}")

            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=20,  # you can increase for deeper search
                scoring="f1",
                cv=5,
                verbose=2,
                random_state=42,
                n_jobs=-1,
            )

            random_search.fit(X_train, y_train)

            logging.info(f"Best params for {model_name}: {random_search.best_params_}")

            # Evaluate
            y_pred = random_search.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            logging.info(
                f"{model_name} Performance -> "
                f"Accuracy: {acc:.4f}, F1: {f1:.4f}, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f}"
            )

            best_model = random_search.best_estimator_
            best_score = f1
            best_model_name = model_name

            logging.info(f"Best Model: {best_model_name} with F1-score: {best_score:.4f}")

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info(f"Saved best model at {self.model_trainer_config.trained_model_file_path}")

            return (
                self.model_trainer_config.trained_model_file_path,
                best_model_name,
                best_score,
            )

        except Exception as e:
            logging.error("Error in initiate_model_trainer")
            raise CustomException(e, sys)

    
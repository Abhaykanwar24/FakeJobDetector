import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os 

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


from src.exception import CustomException
from src.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils import save_object
# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts' , "preprocessor.pkl")



class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        cleaned = []
        for text in X:
            text = str(text).lower()  # lowercase
            words = text.split()
            words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
            cleaned.append(" ".join(words))
        return cleaned

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            numerical_features = ["telecommuting","has_company_logo","has_questions"]
            text_feature = "full_text"

            ## numerical features are already between 0 and 1 no need for standard scaler or any scalling just check null values
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="mean"))
                    
                ]
            )
            logging.info("Creted Numerical pipepline")

            text_pipeline = Pipeline(
                steps=[
                    ("cleaner", TextCleaner()),
                    ("tfidf", TfidfVectorizer(max_features=3000 , ngram_range=(1,2)))
                ]
            )

            logging.info("Creted Text pipepline")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_features),
                    ("text", text_pipeline, text_feature)
                ]
            )
            
            logging.info("Preprocessor pipeline created successfully")

            return preprocessor

        except Exception as e:
            logging.error(f"Error in creating preprocessor: {e}")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")

            # Verify that required columns exist
            required_columns = ["telecommuting", "has_company_logo", "has_questions", "full_text", "fraudulent"]
            for df, name in [(train_df, "train"), (test_df, "test")]:
                if not all(col in df.columns for col in required_columns):
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    raise CustomException(f"Missing columns in {name} data: {missing_cols}", sys)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'fraudulent'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            # Transform input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Convert sparse matrices to dense arrays if necessary
            if hasattr(input_feature_train_arr, 'toarray'):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if hasattr(input_feature_test_arr, 'toarray'):
                input_feature_test_arr = input_feature_test_arr.toarray()

            logging.info(f"Transformed train features shape: {input_feature_train_arr.shape}")
            logging.info(f"Transformed test features shape: {input_feature_test_arr.shape}")

            # Ensure target arrays are 1D
            target_feature_train_arr = np.array(target_feature_train_df).ravel()
            target_feature_test_arr = np.array(target_feature_test_df).ravel()

            # Verify shapes before concatenation
            if input_feature_train_arr.shape[0] != target_feature_train_arr.shape[0]:
                raise CustomException(
                    f"Shape mismatch: input features ({input_feature_train_arr.shape[0]}) "
                    f"and target ({target_feature_train_arr.shape[0]}) for training data", sys
                )
            if input_feature_test_arr.shape[0] != target_feature_test_arr.shape[0]:
                raise CustomException(
                    f"Shape mismatch: input features ({input_feature_test_arr.shape[0]}) "
                    f"and target ({target_feature_test_arr.shape[0]}) for test data", sys
                )

            # Concatenate features and target
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info(f"Final train array shape: {train_arr.shape}")
            logging.info(f"Final test array shape: {test_arr.shape}")

            logging.info("Saved preprocessor object")
            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_tranformation_config.preprocessor_obj_file_path

        except Exception as e:
            logging.error(f"Error in initiate_data_transformation: {e}")
            raise CustomException(e, sys)


import sys
import pandas as pd
from src.exception import CustomException
import os
from src.utils import load_object
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Validate input DataFrame
            expected_columns = ['telecommuting', 'has_company_logo', 'has_questions', 'full_text']
            if not all(col in features.columns for col in expected_columns):
                raise CustomException(f"Input DataFrame missing required columns: {expected_columns}", sys)

            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Apply preprocessing and predict
            data_scaled = preprocessor.transform(features)
            print("Transformed features shape:", data_scaled.shape)
            if hasattr(data_scaled, 'toarray'):
                data_scaled = data_scaled.toarray()
                print("Transformed features sample:", data_scaled[0, :10])  # Log first 10 features
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, telecommuting: int, has_company_logo: int, has_questions: int, full_text: str):
        self.telecommuting = telecommuting
        self.has_company_logo = has_company_logo
        self.has_questions = has_questions
        self.full_text = full_text
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text):
        try:
            # Match TextCleaner in DataTransformation: lowercase, split, remove stopwords, lemmatize
            text = str(text).lower()
            words = text.split()  # Use split() to match DataTransformation
            words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
            return ' '.join(words)
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_as_data_frame(self):
        try:
            # Preprocess full_text
            processed_text = self.preprocess_text(self.full_text)
            custom_data_input_dict = {
                'telecommuting': [self.telecommuting],
                'has_company_logo': [self.has_company_logo],
                'has_questions': [self.has_questions],
                'full_text': [processed_text]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
# 🧠 Fake Job Detector  
**An NLP + Machine Learning Pipeline for Detecting Fake Job Postings**

This project leverages **Natural Language Processing (NLP)** and **Machine Learning** to predict whether a job posting is **genuine or fake**, using only the textual information from the listing.  
It follows a **modular and production-ready architecture** with clear data, training, and prediction pipelines — making it easy to extend and deploy.

---

## 🚀 Project Overview

Fake job postings have become a major concern on online recruitment platforms. This project uses **NLP techniques** and **ML models** to automatically detect such fraudulent postings.

The system:
- Preprocesses and vectorizes job posting text  
- Trains a machine learning model using a balanced dataset (via **SMOTE**)  
- Provides a **prediction pipeline** to evaluate unseen job listings  
- Includes full **EDA**, model training, and **deployment pipeline** for scalability  

---

## 🧩 Key Features

✅ **Modular Programming with Pipelines**  
- `train_pipeline.py`: Handles complete model training, preprocessing, and SMOTE balancing  
- `predict_pipeline.py`: Loads trained model and makes predictions on new job postings  
- `__init__.py`: Initializes the pipeline package for clean imports  

✅ **NLP + Machine Learning**  
- Used NLP techniques to extract meaningful patterns from job descriptions  
- Text vectorization (TF-IDF) for feature extraction  
- Trained classical ML models for reliable and interpretable predictions  

✅ **Data Balancing with SMOTE**  
- Used **SMOTE (Synthetic Minority Oversampling Technique)** to handle class imbalance  
  ```python
  smote = SMOTE(random_state=42)
  X_train, y_train = smote.fit_resample(X_train, y_train)

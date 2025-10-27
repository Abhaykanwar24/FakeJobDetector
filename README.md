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


# ✅ High Model Performance

| Metric | Score |
|--------|--------|
| **Accuracy** | 0.9846 |
| **F1-Score** | 0.8450 |
| **Precision** | 0.9375 |
| **Recall** | 0.7692 |

---

# ✅ EDA and Deployment Ready

- Performed **Exploratory Data Analysis (EDA)** for data understanding and feature engineering  
- Built a **Flask app (`app.py`)** to deploy the trained model and serve real-time predictions  

---

# 🧠 Tech Stack

**Programming Language:**  
- Python 3.10+  

**Libraries & Tools:**  
- `scikit-learn`, `pandas`, `numpy`, `nltk`, `imblearn`, `pickle`, `Flask`  

**Machine Learning:**  
- XGBOOST, Random Forest, and other ensemble methods  

**NLP:**  
- Text cleaning, tokenization, stopword removal, TF-IDF vectorization  

**Deployment:**  
- Flask-based web application  

---

# 🧪 How to Run Locally

### 1️⃣ Clone the Repository
```bash
- git clone https://github.com/Abhaykanwar24/FakeJobDetector.git
- cd FakeJobDetector

###2️⃣ Create and Activate Virtual Environment
- python -m venv venv
- source venv/bin/activate   # For Linux/Mac
- venv\Scripts\activate      # For Windows

###3️⃣ Install Dependencies
pip install -r requirements.txt

###4️⃣ Run the Training Pipeline
- python src/train_pipeline.py

###5️⃣ Run the Flask App
- python app.py




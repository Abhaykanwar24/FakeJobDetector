import streamlit as st
import pandas as pd
import nltk
import os
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# 🧩 Ensure NLTK resources are available (fix for Streamlit deployment)
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.mkdir(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# ✅ Download required NLTK datasets safely
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

# 🎯 Streamlit Page Setup
st.set_page_config(page_title="Fake Job Detector", page_icon="🧠", layout="centered")

# 🧠 Title and Description
st.title("🧠 Fake Job Detector")
st.markdown(
    """
    This app uses **NLP + Machine Learning** to detect whether a job posting is **Legitimate** or **Fraudulent**.  
    Enter the job details below and click **Predict**.
    """
)

# 📝 Input Fields
st.subheader("Enter Job Details")

telecommuting = st.selectbox("Is this a telecommuting job?", ["Select", "0 (No)", "1 (Yes)"])
has_company_logo = st.selectbox("Does the posting have a company logo?", ["Select", "0 (No)", "1 (Yes)"])
has_questions = st.selectbox("Does it include company questions?", ["Select", "0 (No)", "1 (Yes)"])
full_text = st.text_area("Job Description (paste or type the full text here):", height=200)

# 🧮 Prediction Button
if st.button("🔍 Predict"):
    # --- Validation ---
    if "Select" in [telecommuting, has_company_logo, has_questions]:
        st.error("⚠️ Please select valid options for Telecommuting, Company Logo, and Questions.")
    elif not full_text or len(full_text.strip()) < 10:
        st.error("⚠️ Please provide a valid job description (at least 10 characters).")
    else:
        try:
            # --- Prepare Data ---
            data = CustomData(
                telecommuting=int(telecommuting[0]),
                has_company_logo=int(has_company_logo[0]),
                has_questions=int(has_questions[0]),
                full_text=full_text.strip()
            )
            pred_df = data.get_data_as_data_frame()

            # --- Run Prediction ---
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            # --- Map Prediction ---
            label = "🟢 Legitimate Job Posting" if results[0] == 0 else "🔴 Fraudulent Job Posting"

            # --- Display Result ---
            st.success(f"**Prediction Result:** {label}")
            st.markdown("---")
            st.write("### ✅ Input Summary")
            st.dataframe(pred_df)

        except Exception as e:
            st.error(f"❌ An error occurred during prediction:\n\n{str(e)}")

# 🧾 Footer
st.markdown("---")
st.markdown(
    "Developed by **Abhaykanwar Singh** | "
    "[GitHub](https://github.com/Abhaykanwar24) | "
)
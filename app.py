from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        try:
            # Collect and validate form data
            telecommuting = request.form.get('telecommuting')
            has_company_logo = request.form.get('has_company_logo')
            has_questions = request.form.get('has_questions')
            full_text = request.form.get('full_text')

            # Validate inputs
            if not all([telecommuting in ['0', '1'], has_company_logo in ['0', '1'], has_questions in ['0', '1']]):
                return render_template('home.html', error="Please select valid options for Telecommuting, Company Logo, and Questions.")
            if not full_text or len(full_text.strip()) < 10:
                return render_template('home.html', error="Please provide a valid job description (at least 10 characters).")

            # Convert to integers
            data = CustomData(
                telecommuting=int(telecommuting),
                has_company_logo=int(has_company_logo),
                has_questions=int(has_questions),
                full_text=full_text
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            print("Input DataFrame:", pred_df)

            # Run prediction pipeline
            predict_pipeline = PredictPipeline()
            print("Starting Prediction")
            results = predict_pipeline.predict(pred_df)
            print("Prediction Result:", results)

            # Map prediction to label (0 = Legitimate, 1 = Fraudulent)
            label = "Legitimate" if results[0] == 0 else "Fraudulent"

            # Render template with results and form values
            return render_template(
                'home.html',
                results=label,
                prediction=True,
                telecommuting=telecommuting,
                has_company_logo=has_company_logo,
                has_questions=has_questions,
                full_text=full_text
            )

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return render_template('home.html', error=f"An error occurred: {str(e)}. Please try again.")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
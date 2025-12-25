# Diabetes Prediction Model

## Live Demo
You can try the deployed Streamlit app here: [https://invalid3rror-diabetes-ml-quest-app.streamlit.app/](https://invalid3rror-diabetes-ml-quest-app.streamlit.app/)

## Overview
This project uses a machine learning model to predict the likelihood of diabetes in patients based on medical data. The model is built using an open-source dataset from the National Institute of Diabetes and Digestive and Kidney Diseases, available on Kaggle.

## Dataset
- **Source:** Kaggle
- **Name:** Pima Indians Diabetes Database
- **File:** diabetes.csv
- **Features:**
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (1 = diabetes, 0 = no diabetes)

## Model Workflow
1. **Data Loading:** The dataset is loaded from diabetes.csv.
2. **Preprocessing:** Data is cleaned and prepared for training (e.g., handling missing values, scaling features).
3. **Training:** A machine learning algorithm (commonly Logistic Regression, Random Forest, or similar) is trained to classify patients as diabetic or not.
4. **Evaluation:** The model's accuracy and performance are evaluated using metrics like accuracy, precision, recall, and F1-score.
5. **Prediction:** The trained model predicts diabetes risk for new patient data.
6. **Deployment:** The model is deployed using Streamlit, providing an interactive web interface for users to input patient data and receive predictions.

## Steps to Clean and Prepare the CSV (Expanded)
1. **Load the Data:**
   - Use Pandas to read diabetes.csv into a DataFrame.
2. **Explore the Data:**
   - Print info, summary statistics, and check for missing values.
   - Visualize feature distributions and correlations using matplotlib and seaborn.
3. **Identify Missing or Invalid Values:**
   - In columns like Glucose, BloodPressure, SkinThickness, Insulin, and BMI, zeros are treated as missing values.
4. **Replace Zeros and Impute:**
   - Replace zeros with NaN, then fill missing values with the median of each column for robustness.
5. **Feature Engineering:**
   - Optionally, use PolynomialFeatures to create interaction terms and improve model performance.
6. **Split Features and Target:**
   - Separate the DataFrame into features (X) and target (y).
7. **Feature Scaling:**
   - Use StandardScaler to normalize features, ensuring consistent scale for model training.
8. **Train-Test Split:**
   - Split the data into training and testing sets to evaluate model performance.
9. **Model Training:**
   - Train models such as Logistic Regression, Decision Tree, and Random Forest. Use pipelines for streamlined preprocessing and training.
10. **Export Model and Scaler:**
    - Save the trained model and scaler using joblib for deployment in the Streamlit app.

These steps are based on the workflow in model.ipynb and ensure the data is clean, well-prepared, and suitable for building a reliable diabetes prediction model.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Use the web interface to input patient data and view predictions.

## References
- [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- National Institute of Diabetes and Digestive and Kidney Diseases

---
This project demonstrates a practical application of machine learning in healthcare for early diabetes detection.
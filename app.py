import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight:bold;
}
.stSlider {
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Title and description
st.title("🩺 Diabetes Risk Prediction")
st.markdown("### Assess Your Diabetes Risk Factors")

# Create columns for input
col1, col2 = st.columns(2)

with col1:
    # Numerical inputs with sliders
    pregnancies = st.slider("Number of Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level", 0, 250, 100)
    blood_pressure = st.slider("Blood Pressure", 0, 150, 70)
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)

with col2:
    # More numerical inputs with sliders
    insulin = st.slider("Insulin Level", 0, 1000, 80)
    bmi = st.slider("Body Mass Index (BMI)", 0.0, 70.0, 25.0, step=0.1)
    diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.0, 0.5, step=0.01)
    age = st.slider("Age", 0, 100, 30)

# Risk factor checkboxes
st.markdown("### Additional Risk Factors")
col3, col4 = st.columns(2)

with col3:
    family_history = st.checkbox("Family History of Diabetes")
    overweight = st.checkbox("Overweight")

with col4:
    sedentary_lifestyle = st.checkbox("Sedentary Lifestyle")
    high_stress = st.checkbox("High Stress Levels")

# Predict button
if st.button("Predict Diabetes Risk", type="primary"):
    # Prepare input data
    input_data = np.array([
        pregnancies, glucose, blood_pressure, 
        skin_thickness, insulin, bmi, 
        diabetes_pedigree, age
    ]).reshape(1, -1)
    
    # Predict
    prediction = model.predict_proba(input_data)[0][1]
    
    # Adjust prediction based on additional risk factors
    risk_multipliers = [
        1.2 if family_history else 1.0,
        1.3 if overweight else 1.0,
        1.2 if sedentary_lifestyle else 1.0,
        1.1 if high_stress else 1.0
    ]
    
    adjusted_risk = prediction * np.prod(risk_multipliers)
    adjusted_risk = min(adjusted_risk, 1.0)  # Cap at 100%
    
    # Display results
    st.markdown(f"### Diabetes Risk Prediction")
    
    # Risk level classification
    if adjusted_risk < 0.3:
        risk_level = "Low Risk"
        color = "green"
    elif adjusted_risk < 0.6:
        risk_level = "Moderate Risk"
        color = "orange"
    else:
        risk_level = "High Risk"
        color = "red"
    
    # Results display
    st.markdown(f"<p class='big-font' style='color:{color};'>Risk Level: {risk_level}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='big-font'>Estimated Diabetes Risk: {adjusted_risk*100:.2f}%</p>", unsafe_allow_html=True)
    
    # Recommendations based on risk
    st.markdown("### Recommendations")
    if risk_level == "Low Risk":
        st.success("Continue maintaining your healthy lifestyle!")
    elif risk_level == "Moderate Risk":
        st.warning("Consider consulting a healthcare professional and making lifestyle changes.")
    else:
        st.error("Strongly recommend consulting a healthcare professional for comprehensive assessment.")

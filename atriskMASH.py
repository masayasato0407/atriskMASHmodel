import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load('./random_forest_model.pkl')

model = load_model()

st.title('STEALTH-ARMS (STEALTH study-derived At-Risk MASH Stratification) model')
st.markdown("URL of the original article: To be determined")

st.header('Input variables and press the predict button')
age = st.number_input('Patient age', min_value=18, max_value=100, value=50)
DM = 1 if st.radio('Diabetes', ['Absent', 'Present']) == 'Present' else 0
HTN = 1 if st.radio('Hypertension', ['Absent', 'Present']) == 'Present' else 0
AST = st.number_input('AST (U/L)', min_value=1, max_value=300, value=30)
GGT = st.number_input('γ-GTP (U/L)', min_value=1, max_value=1000, value=30)
Plt = st.number_input('Platelet count ( × 104/µL)', min_value=1.0, max_value=75.0, value=20.0,step=0.1,format="%.1f")
INR = st.number_input('PT-INR', min_value=0.50, max_value=3.00, value=1.00, step=0.01, format="%.2f")

input_data = pd.DataFrame({
    'age': [age],
    'DM': [DM],
    'HTN': [HTN],
    'AST': [AST],
    'GGT': [GGT],
    'Plt': [Plt],
    'INR': [INR]
})

if st.button('Predict'):
    model = load_model()
    probability = model.predict_proba(input_data)[0][1]
    
    st.header('Prediction Result')
    st.markdown(f'<h3 style="font-size: 20px;">Probability of having "at-risk MASH" in this patient = {probability:.2%}</h3>', unsafe_allow_html=True)

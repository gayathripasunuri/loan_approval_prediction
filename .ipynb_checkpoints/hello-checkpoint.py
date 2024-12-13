import streamlit as st
import pandas as pd
import pickle as pk

# Load the model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

st.header('Loan Prediction App')

# User input for features
no_od_dep = st.slider('Choose No of dependents', 0, 5)
grad = st.selectbox('Choose Education', ['Graduate', 'Not Graduate'])
self_emp = st.selectbox('Self Employed?', ['Yes', 'No'])
Annual_Income = st.slider('Choose Annual Income', 0, 10000000)
Loan_Amount = st.slider('Choose Loan Amount', 0, 10000000)
Loan_Dur = st.slider('Choose Loan Duration (in years)', 0, 20)
cibil = st.slider('Choose Cibil Score', 0, 1000)  # Changed to slider
Assets = st.slider('Choose Assets', 0, 10000000)

# Encoding education and self-employed status
if grad == 'Graduate':
    grad_s = 1
else:
    grad_s = 0

if self_emp == 'No':
    emp_s = 0
else:
    emp_s = 1

# Prepare data for prediction
if st.button("Predict"):
    pred_data = pd.DataFrame([[
        no_od_dep, grad_s, emp_s, Annual_Income, Loan_Amount, Loan_Dur, cibil, Assets
    ]], columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets'])

    # Scale data and make prediction
    pred_data_scaled = scaler.transform(pred_data)
    prediction = model.predict(pred_data_scaled)

    # Display result
    if prediction[0] == 1:
        st.markdown('**Loan is Approved**')
    else:
        st.markdown('**Loan is Rejected**')

# Update pip
subprocess.run(['pip', 'install'])

# Install required packages
subprocess.run(['pip', 'install', 'keras==2.15.0'])


import pickle
import pandas as pd
import numpy as np
import streamlit as st

# Load the pickled model
model = pickle.load(open('churn_model.pkl', 'rb'))

# Define the function to make predictions
def predict_churn(data):
  # Convert data to a DataFrame
  data = pd.DataFrame(data)

  # Standardize the data
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(data)

  # Make predictions
  predictions = model.predict(scaled_data)

  # Convert predictions to probabilities
  probabilities = model.predict_proba(scaled_data)

  return predictions, probabilities

# Create the Streamlit app
st.title('Churn Prediction App')

# Collect user input
st.header('Customer Information')
gender = st.selectbox('Gender', ['Male', 'Female'])
senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.number_input('Tenure')
internet_service = st.selectbox('Internet Service', ['No', 'DSL', 'Fiber Optic'])
online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
monthly_charges = st.number_input('Monthly Charges')
total_charges = st.number_input('Total Charges')

# Prepare the data for prediction
data = {
  'gender': gender,
  'senior_citizen': senior_citizen,
  'partner': partner,
  'dependents': dependents,
  'tenure': tenure,
  'internet_service': internet_service,
  'online_security': online_security,
  'online_backup': online_backup,
  'device_protection': device_protection,
  'tech_support': tech_support,
  'streaming_movies': streaming_movies,
  'streaming_tv': streaming_tv,
  'monthly_charges': monthly_charges,
  'total_charges': total_charges
}

# Make predictions
predictions, probabilities = predict_churn(data)

# Display the predictions
if predictions == 0:
  st.success('Customer is not likely to churn')
else:
  st.error('Customer is likely to churn')

# Display the probabilities
st.subheader('Prediction Probabilities')
st.write('Churn: {:.2f}'.format(probabilities[0][1]))
st.write('Not Churn: {:.2f}'.format(probabilities[0][0]))

import subprocess
subprocess.call(['pip', 'install', 'keras==2.15.0', 'scikit-learn==0.24.0'])
import sklearn


from sklearn.preprocessing import LabelEncoder, StandardScaler
label_encoder = LabelEncoder()

# churn_app.py
import streamlit as st
import joblib

# Specify the path to your .joblib file
model_filename = 'churn_model.joblib'

# Load the model from the specified file
model_y = joblib.load(model_filename)

# Streamlit app
def main():
    st.title("Churn Prediction App")
    st.write("Enter customer details to predict churn.")

    # User input
    gender_options = ['Male', 'Female']
    partner_options = ['No', 'Yes']
    dependents_options = ['No', 'Yes']
    phone_service_options = ['Yes', 'No']
    multiple_lines_options = ['No', 'No phone service', 'Yes']
    internet_service_options = ['DSL', 'Fiber optic', 'No']
    online_security_options = ['No', 'Yes', 'No internet service']
    online_backup_options = ['No', 'Yes', 'No internet service']
    device_protection_options = ['No', 'Yes', 'No internet service']
    tech_support_options = ['No', 'Yes', 'No internet service']
    senior_citizen_options = ["Yes", "No"]
    streaming_tv_options = ['No', 'No internet service', 'Yes']
    streaming_movies_options = ['No', 'Yes', 'No internet service']
    contract_options = ['Month-to-month', 'One year', 'Two year']
    paperless_billing_options = ['No', 'Yes']
    payment_method_options = ['Electronic check', 'Mailed check','Bank transfer (automatic)','Credit card (automatic)']

    gender = st.selectbox('Gender', gender_options)
    partner = st.selectbox('Partner', partner_options)
    dependents = st.selectbox('Dependents', dependents_options)
    phone_service = st.selectbox('Phone Service', phone_service_options)
    multiple_lines = st.selectbox('Multiple Lines', multiple_lines_options)
    internet_service = st.selectbox('Internet Service', internet_service_options)
    online_security = st.selectbox('Online Security', online_security_options)
    online_backup = st.selectbox('Online Backup', online_backup_options)
    device_protection = st.selectbox('Device Protection', device_protection_options)
    tech_support = st.selectbox('Tech Support', tech_support_options)
    senior_citizen = st.selectbox('SeniorCitizen', senior_citizen_options)
    streaming_tv = st.selectbox('Streaming TV', streaming_tv_options)
    streaming_movies = st.selectbox('Streaming Movies', streaming_movies_options)
    contract = st.selectbox('Contract', contract_options)
    paperless_billing = st.selectbox('Paperless Billing', paperless_billing_options)
    payment_method = st.selectbox('Payment Method', payment_method_options)

    # Numeric inputs
    monthly_charges = st.number_input('Monthly Charges', min_value=0)
    total_charges = st.number_input('Total Charges', min_value=0)
    tenure = st.number_input('Tenure (months)', min_value=0)

    # Convert user input to a DataFrame
    user_input = {
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'SeniorCitizen': senior_citizen,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'tenure': tenure,
    }

    input_df = pd.DataFrame([user_input])
    
    numeric = input_df.iloc[:,17:19]
    numeric["tenure"] = input_df["tenure"]
    non_numeric = input_df.drop(["tenure", "MonthlyCharges", "TotalCharges"], axis = 1)
    
    for column in non_numeric.columns:
        if non_numeric[column].dtype == 'object':  # Check if the column is of object type (categorical)
            non_numeric[column] = label_encoder.fit_transform(non_numeric[column])
        
    non_numeric = non_numeric.drop(["StreamingTV", "StreamingMovies", "PhoneService"], axis=1)
    
    x = pd.concat([non_numeric, numeric], axis=1)
    print(x.shape)

    x.head(2)
    # Make predictions
    prediction = model_y.predict(x)


    # Display the prediction
    st.write("Churn Prediction:")
    st.write(prediction)

if __name__ == '__main__':
    main()

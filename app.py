import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Load the scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

# streamlit app
st.title('Customer Churn Prediction')

#user inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', min_value=0.0, max_value=100000.0, value=5000.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, max_value=200000.0, value=50000.0)
tenure = st.slider('Tenure (Years)', 0, 10, 1)
number_of_products = st.slider('Number of Products', 1, 4, 1)
has_credit_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out())
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

# Display the prediction
if prediction_prob > 0.5:
    st.write(f'Customer is likely to churn with a probability of {prediction_prob:.2f}')
else:
    st.write(f'Customer is likely to stay with a probability of {1 - prediction_prob:.2f}')
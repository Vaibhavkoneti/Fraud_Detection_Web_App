import streamlit as st
from Model_utils import DNN, load_model
import torch
from sklearn.preprocessing import StandardScaler

input_size = 7 
model = DNN(input_size)
model = load_model(model, "Fraud_Dectection_model.pth")

scaler = StandardScaler()

# Set page configuration
st.set_page_config(page_title="Fraud Detection App", page_icon=":shield:", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
# Dropdown navigation
page = st.sidebar.selectbox("Go to", ["Home", "About", "Features"])



if page == "Home":
    st.title("🛡️ Fraud Detection App")
    st.write("Detect fraudulent transactions with ease!")

    # Additional features
    year = st.slider('Transaction Year', min_value=2010, max_value=2023, value=2022)
    month = st.slider('Transaction Month', min_value=1, max_value=12, value=6)
    merchant_name = st.text_input('Merchant Name', 'Sample Merchant')
    merchant_city = st.text_input('Merchant City', 'Sample City')
    mcc = st.number_input('MCC (Merchant Category Code)', min_value=1000, max_value=9999, value=5000)
    hours = st.slider('Transaction Hours', min_value=0, max_value=23, value=12)
    minute = st.slider('Transaction Minute', min_value=0, max_value=59, value=30)

    # Feature to indicate time of day (morning, afternoon, evening, night)
    time_of_day = 'Morning' if 6 <= hours < 12 else 'Afternoon' if 12 <= hours < 18 else 'Evening' if 18 <= hours < 24 else 'Night'

    # Display additional features
    st.write(f"**Time of Day:** {time_of_day}")

    # Predict button
    if st.button('Predict Fraud'):
        # Preprocess user input
        user_input = torch.tensor([[year, month, merchant_name, merchant_city, mcc, hours, minute]], dtype=torch.float32)
        user_input[:, :5] = scaler.fit_transform(user_input[:, :5])  # Use fit_transform for training data
        user_input[:, 5:] = user_input[:, 5:].astype(float)

        # Make prediction
        with torch.no_grad():
            model.eval()
            prediction = model(user_input)

        # Display result
        st.success('Prediction: ' + ':lock:' if prediction.item() >= 0.5 else ':thumbsup:')

elif page == "About":
    st.title("About")
    st.write("This app is designed to detect fraudulent transactions using machine learning.")
    st.write("It is powered by a Deep Neural Network trained on historical transaction data.")

elif page == "Features":
    st.title("Features")
    st.write("1. Predict fraud based on transaction details.")
    st.write("2. Explore additional features and time of day information.")
    st.write("3. Access the 'About' section for more information.")

# Add a subtitle
st.markdown("<h2 style='text-align: center; color: #004d99;'>Fraud Detection Web App</h2>", unsafe_allow_html=True)

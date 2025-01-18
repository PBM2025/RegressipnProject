import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO

# Load the trained models and scaler
@st.cache_resource
def load_models():
    scaler = StandardScaler()
    lin_model = LinearRegression()
    dt_model = DecisionTreeRegressor(random_state=42)
    return scaler, lin_model, dt_model

# Streamlit App Title
st.title("Car Price Prediction App")
st.write("Upload a dataset to predict car prices using trained models.")

# Upload dataset for training
uploaded_train_file = st.file_uploader("Upload Training Dataset (Excel file)", type=["xlsx", "xls"])

if uploaded_train_file:
    # Load dataset
    train_data = pd.read_excel(uploaded_train_file)
    st.subheader("Training Dataset Preview")
    st.write(train_data.head())

    # Handle missing values
    for col in train_data.select_dtypes(include=np.number).columns:
        train_data[col].fillna(train_data[col].median(), inplace=True)
    for col in train_data.select_dtypes(include=['object']).columns:
        train_data[col].fillna(train_data[col].mode()[0], inplace=True)

    # One-hot encoding for categorical columns
    train_data = pd.get_dummies(train_data, drop_first=True)

    # Define features and target
    X_train = train_data.drop('Price', axis=1)
    y_train = train_data['Price']

    # Scale the features
    scaler, lin_model, dt_model = load_models()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train the Linear Regression Model
    lin_model.fit(X_train_scaled, y_train)

    # Train the Decision Tree Model
    dt_model.fit(X_train_scaled, y_train)

    st.success("Models trained successfully!")

# Upload new data for prediction
uploaded_test_file = st.file_uploader("Upload Data for Prediction (Excel file)", type=["xlsx", "xls"])

if uploaded_test_file:
    # Load the test dataset
    test_data = pd.read_excel(uploaded_test_file)
    st.subheader("New Data Preview")
    st.write(test_data.head())

    try:
        # Preprocess the test dataset
        test_data = pd.get_dummies(test_data, drop_first=True)

        # Ensure the test dataset matches the training dataset columns
        missing_cols = set(X_train.columns) - set(test_data.columns)
        for col in missing_cols:
            test_data[col] = 0
        test_data = test_data[X_train.columns]  # Reorder columns to match training data

        # Scale the test data
        test_data_scaled = scaler.transform(test_data)

        # Generate predictions
        lin_predictions = lin_model.predict(test_data_scaled)
        dt_predictions = dt_model.predict(test_data_scaled)

        # Add predictions to the test dataset
        test_data['Linear Regression Prediction'] = lin_predictions
        test_data['Decision Tree Prediction'] = dt_predictions

        # Display predictions
        st.subheader("Predictions")
        st.write(test_data[['Linear Regression Prediction', 'Decision Tree Prediction']])

        # Prepare predictions for download as Excel
        output_buffer = BytesIO()
        with pd.ExcelWriter(output_buffer, engine='xlsxwriter') as writer:
            test_data.to_excel(writer, index=False)
        output_buffer.seek(0)

        st.download_button(
            label="Download Predictions as Excel",
            data=output_buffer,
            file_name="car_price_predictions.xlsx",
            mime="application/vnd.ms-excel",
        )

    except Exception as e:
        st.error(f"Error processing the new data: {e}")

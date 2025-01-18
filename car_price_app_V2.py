import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO

# Streamlit App Title
st.title("Car Price Prediction and Model Analysis App")
st.write("This app performs EDA, trains models, and predicts car prices.")

# Upload dataset for EDA and training
uploaded_file = st.file_uploader("Upload Dataset (Excel file)", type=["xlsx", "xls"])

if uploaded_file:
    # Load dataset
    df = pd.read_excel(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Summary statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    # Check for missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numerical_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Distribution of Price
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Price'], kde=True, color='blue', ax=ax)
    ax.set_title("Distribution of Car Prices")
    st.pyplot(fig)

    # Handle missing values
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # One-hot encoding for categorical columns
    df = pd.get_dummies(df, drop_first=True)

    # Define features and target
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the Linear Regression Model
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_pred_lin = lin_model.predict(X_test)
    lin_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lin))
    lin_r2 = r2_score(y_test, y_pred_lin)

    # Train the Decision Tree Model
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    dt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt))
    dt_r2 = r2_score(y_test, y_pred_dt)

    # Display model performance metrics
    st.subheader("Model Performance Metrics")
    st.write(f"**Linear Regression**: RMSE = {lin_rmse:.2f}, R² = {lin_r2:.2f}")
    st.write(f"**Decision Tree**: RMSE = {dt_rmse:.2f}, R² = {dt_r2:.2f}")

    # Compare models
    st.subheader("Model Comparison")
    model_comparison = pd.DataFrame({
        'Model': ['Linear Regression', 'Decision Tree'],
        'RMSE': [lin_rmse, dt_rmse],
        'R²': [lin_r2, dt_r2]
    })
    st.write(model_comparison)

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
            missing_cols = set(X.columns) - set(test_data.columns)
            for col in missing_cols:
                test_data[col] = 0
            test_data = test_data[X.columns]  # Reorder columns to match training data

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

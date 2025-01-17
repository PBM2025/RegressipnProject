import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Title and description
st.title("Car Price Prediction App")
st.write("""
This app performs the following tasks:
- Upload car price dataset (Excel file).
- Perform EDA (Descriptive Statistics and Visualizations).
- Preprocess the data (Handle missing values, scaling, encoding).
- Train Multiple Linear Regression and Decision Tree Regression models.
- Allow the user to upload new data (Excel file) for predictions.
""")

# Upload dataset
uploaded_file = st.file_uploader("Upload a Car Price Dataset (Excel file)", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_excel(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(data.head())

    # EDA: Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

    # EDA: Visualizations
    st.subheader("Data Visualizations")
    
    # Heatmap for correlations
    st.write("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Distribution of Price
    st.write("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data['Price'], kde=True, ax=ax, color="blue")
    st.pyplot(fig)

    # Data Preprocessing
    st.subheader("Data Preprocessing")

    # Handle missing values
    if data.isnull().sum().sum() > 0:
        st.write("Missing Values Found:")
        st.write(data.isnull().sum())
        data.fillna(data.median(), inplace=True)
        st.write("Missing values filled with median.")

    # Encoding categorical variables
    data = pd.get_dummies(data, drop_first=True)
    st.write("After Encoding:")
    st.write(data.head())

    # Feature Scaling
    scaler = StandardScaler()
    X = data.drop("Price", axis=1)
    y = data["Price"]
    X_scaled = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build Models
    st.subheader("Model Training")

    # Multiple Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_pred = lin_model.predict(X_test)
    lin_rmse = np.sqrt(mean_squared_error(y_test, lin_pred))
    st.write(f"Linear Regression RMSE: {lin_rmse:.2f}")

    # Decision Tree Regression
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
    st.write(f"Decision Tree RMSE: {dt_rmse:.2f}")

    # Prediction on New Data
    st.subheader("Make Predictions on New Data")

    # Upload new data for prediction
    new_data_file = st.file_uploader("Upload New Data (Excel file)", type=["xlsx", "xls"])

    if new_data_file is not None:
        new_data = pd.read_excel(new_data_file)
        st.write("New Data Preview:")
        st.write(new_data.head())

        # Ensure the new data has the same columns as training data
        try:
            new_data = pd.get_dummies(new_data, drop_first=True)
            missing_cols = set(X.columns) - set(new_data.columns)
            for col in missing_cols:
                new_data[col] = 0
            new_data = new_data[X.columns]  # Reorder columns to match training data
            new_data_scaled = scaler.transform(new_data)

            # Predict prices using the linear regression model
            new_data_predictions = lin_model.predict(new_data_scaled)
            new_data["Predicted Price"] = new_data_predictions
            st.write("Predictions:")
            st.write(new_data)

            # Download predictions as an Excel file
            st.download_button(
                label="Download Predictions as Excel",
                data=new_data.to_excel(index=False),
                file_name="predicted_car_prices.xlsx",
                mime="application/vnd.ms-excel",
            )
        except Exception as e:
            st.error(f"Error processing new data: {e}")

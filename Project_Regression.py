import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# App Title
st.title("Car Price Prediction App")

# Upload Dataset
uploaded_file = st.file_uploader("Upload a Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset")
    st.write(data.head())

    # Exploratory Data Analysis
    st.subheader("Exploratory Data Analysis")
    st.write("Summary Statistics:")
    st.write(data.describe())

    st.write("Correlation Heatmap:")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Data Preprocessing
    st.subheader("Data Preprocessing")

    # Check for missing values
    if data.isnull().sum().sum() > 0:
        st.write("Missing Values Found!")
        st.write(data.isnull().sum())
        data.fillna(data.median(), inplace=True)
        st.write("Missing values filled with median.")

    # Encode categorical variables
    data = pd.get_dummies(data, drop_first=True)
    st.write("After Encoding:")
    st.write(data.head())

    # Feature Scaling
    scaler = StandardScaler()
    X = data.drop("Price", axis=1)
    y = data["Price"]
    X_scaled = scaler.fit_transform(X)

    # Split the data
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

    # Allow Users to Input New Data
    st.subheader("Make Predictions")

    user_input = {}
    for col in X.columns:
        user_input[col] = st.text_input(f"Enter value for {col}:")

    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([user_input], columns=X.columns)
            input_scaled = scaler.transform(input_df)
            prediction = lin_model.predict(input_scaled)
            st.success(f"Predicted Price: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Housing Price Prediction App")

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target

st.subheader("Dataset Preview")
st.dataframe(df.head())

X = df[["MedInc", "AveRooms", "HouseAge"]]
y = df["Price"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write("Mean Squared Error:", mse)
st.write("R2 Score:", r2)

st.subheader("Actual vs Predicted Plot")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted Housing Prices")
st.pyplot(fig)

st.subheader("Try Your Own Prediction")

income = st.slider("Median Income", float(df.MedInc.min()), float(df.MedInc.max()), 3.0)
rooms = st.slider("Average Rooms", float(df.AveRooms.min()), float(df.AveRooms.max()), 5.0)
age = st.slider("House Age", float(df.HouseAge.min()), float(df.HouseAge.max()), 20.0)

user_input = scaler.transform([[income, rooms, age]])
prediction = model.predict(user_input)

st.write("Predicted House Price:", float(prediction[0]))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

st.title("🚰 Smart Community Water Monitoring Dashboard")

# Generate dataset
houses = [f"H{i}" for i in range(1,21)]

timestamps = pd.date_range(
    start="2026-01-01",
    periods=30*24,
    freq="H"
)

data = []

for house in houses:

    base = np.random.randint(5,15)

    for time in timestamps:

        hour = time.hour

        if 6 <= hour <= 9 or 18 <= hour <= 21:
            usage = base + np.random.randint(5,15)
        else:
            usage = base + np.random.randint(0,5)

        if np.random.rand() < 0.001:
            usage += np.random.randint(50,100)

        data.append([time, house, usage])

df = pd.DataFrame(
    data,
    columns=["Timestamp","House_ID","Water_Liters"]
)

# Sidebar house selector
house = st.sidebar.selectbox(
    "Select House",
    df["House_ID"].unique()
)

# Community usage
st.header("Community Water Consumption")

community_usage = df.groupby("House_ID")["Water_Liters"].mean().reset_index()

fig = px.bar(
    community_usage,
    x="House_ID",
    y="Water_Liters"
)

st.plotly_chart(fig)

# House analysis
st.header("House Water Usage")

house_data = df[df["House_ID"] == house]

fig = px.line(
    house_data,
    x="Timestamp",
    y="Water_Liters"
)

st.plotly_chart(fig)

# Peak usage
st.header("Peak Usage Hours")

df["Hour"] = df["Timestamp"].dt.hour

hourly_usage = df.groupby("Hour")["Water_Liters"].mean().reset_index()

fig = px.bar(
    hourly_usage,
    x="Hour",
    y="Water_Liters"
)

st.plotly_chart(fig)

# Leak detection
st.header("Leak Detection")

model = IsolationForest(contamination=0.02)

house_data["anomaly"] = model.fit_predict(
    house_data[["Water_Liters"]]
)

fig = px.scatter(
    house_data,
    x="Timestamp",
    y="Water_Liters",
    color="anomaly"
)

st.plotly_chart(fig)

# Forecasting
st.header("Water Usage Forecast")

house_data = house_data.reset_index()

house_data["time_index"] = np.arange(len(house_data))

model = LinearRegression()

model.fit(
    house_data[["time_index"]],
    house_data["Water_Liters"]
)

future = np.arange(len(house_data), len(house_data)+24).reshape(-1,1)

prediction = model.predict(future)

forecast_df = pd.DataFrame({
    "Future_Hour": range(1,25),
    "Predicted_Usage": prediction
})

fig = px.line(
    forecast_df,
    x="Future_Hour",
    y="Predicted_Usage"
)

st.plotly_chart(fig)
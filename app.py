import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("aapl_lstm_model.h5")
scaler = joblib.load("scaler.pkl")

st.title("AAPL Stock Close Price Prediction (Next 30 Days)")
st.write("Predicts future closing prices using Trained LSTM Model")

uploaded = st.file_uploader("Upload AAPL.csv file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df["Date"] = pd.to_datetime(df["Date"])
    
    data = df["Close"].values.reshape(-1, 1)
    scaled_data = scaler.transform(data)

    look_back = 60
    last_sequence = scaled_data[-look_back:].reshape(1, look_back, 1)
    future_predictions_scaled = []

    future_days = 30
    for _ in range(future_days):
        next_scaled = model.predict(last_sequence, verbose=0)
        future_predictions_scaled.append(next_scaled[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], [[next_scaled[0]]], axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=future_days+1, freq='B')[1:]

    future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions.flatten()})

    st.subheader("📌 Next 30 Business Days Prediction")
    st.dataframe(future_df)

    st.line_chart(future_df.set_index("Date"))

    csv = future_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions CSV", csv, "AAPL_30day_predictions.csv", "text/csv")

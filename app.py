import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import time

# Streamlit Header
st.header('📈 Stock Market Predictor')

# Load the trained model
model = load_model(r'C:\Users\rajat\OneDrive\Desktop\STOCK\Stock Predictions Model.keras')

# User Input for Stock Symbol
stock = st.text_input('Enter Stock Symbol (e.g., AAPL, GOOG, TATAMOTORS.NS)', 'GOOG')

start = '2012-01-01'
end = '2022-12-31'

# Function to Fetch Stock Data with Retry
def fetch_stock_data(ticker, max_retries=5, wait_time=5):
    for i in range(max_retries):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if not data.empty:
                return data
            st.warning(f"Attempt {i+1}: No data retrieved, retrying...")
        except Exception as e:
            st.warning(f"Attempt {i+1} failed: {e}")
        time.sleep(wait_time)
    return None

# Fetch Stock Data
data = fetch_stock_data(stock)
if data is None:
    st.error("❌ Failed to fetch stock data. Please try a different symbol or try again later.")
    st.stop()

# Display Data
st.subheader('🔢 Raw Stock Data')
st.write(data.tail())

# Split Data into Training & Testing
data_train = pd.DataFrame(data['Close'][:int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

# Scaling Data
scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

# Ensure `data_test` is not empty before applying `MinMaxScaler`
if data_test.empty:
    st.error("Error: `data_test` is empty after processing. Check stock symbol and data availability.")
    st.stop()

# Scale Data
data_test_scaled = scaler.fit_transform(data_test)

# Moving Averages Plots
st.subheader('📉 Price vs MA50')
ma_50 = data['Close'].rolling(50).mean()

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(data['Close'], label='Close Price')
ax1.plot(ma_50, label='50-Day MA', color='orange')
ax1.set_title('Price vs 50-Day Moving Average')
ax1.set_xlabel('Time')
ax1.set_ylabel('Price')
ax1.legend()
st.pyplot(fig1)

st.subheader('📊 Price vs MA50 & MA100')
ma_100 = data['Close'].rolling(100).mean()

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(data['Close'], label='Close Price')
ax2.plot(ma_50, label='50-Day MA', color='orange')
ax2.plot(ma_100, label='100-Day MA', color='blue')
ax2.set_title('Price vs 50 & 100-Day Moving Averages')
ax2.set_xlabel('Time')
ax2.set_ylabel('Price')
ax2.legend()
st.pyplot(fig2)

st.subheader('📈 Price vs MA100 & MA200')
ma_200 = data['Close'].rolling(200).mean()

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(data['Close'], label='Close Price')
ax3.plot(ma_100, label='100-Day MA', color='blue')
ax3.plot(ma_200, label='200-Day MA', color='red')
ax3.set_title('Price vs 100 & 200-Day Moving Averages')
ax3.set_xlabel('Time')
ax3.set_ylabel('Price')
ax3.legend()
st.pyplot(fig3)

# Prepare Data for Model Prediction
x_test = []
y_test = []

for i in range(100, len(data_test_scaled)):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# Ensure x_test has valid data
if x_test.shape[0] == 0:
    st.error("❌ Not enough data for model prediction. Please try a stock with more historical data.")
    st.stop()

# Make Predictions
predicted_prices = model.predict(x_test)

# Inverse Transform
predicted_prices = scaler.inverse_transform(predicted_prices)
y_test = y_test.reshape(-1, 1)
actual_prices = scaler.inverse_transform(y_test)

# Plot Predicted vs Actual Prices
st.subheader('📈 Predicted vs Actual Closing Prices')
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(actual_prices, label='Actual Price', color='green')
ax4.plot(predicted_prices, label='Predicted Price', color='red')
ax4.set_title('Predicted vs Actual Closing Prices')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
ax4.legend()
st.pyplot(fig4)

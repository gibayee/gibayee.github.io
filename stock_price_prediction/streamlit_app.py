import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import yfinance as yf  # For fetching real-time stock data

# Title of the app
st.title("Real-Time Stock Price Prediction using LSTM")

# Note for users about data loading time
st.warning("Since the model fetches real-time data, it may take about 2 minutes to load depending on your internet connection. Please be patient.")

# Step 1: Allow user to enter any stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", value="AAPL").upper()

# Fetch real-time stock data using yfinance
@st.cache_data
def fetch_stock_data(ticker, period="5y"):
    stock_data = yf.download(ticker, period=period)
    if stock_data.empty:
        st.error(f"No data found for ticker symbol: {ticker}")
        return None
    return stock_data

if ticker:
    stock_data = fetch_stock_data(ticker)
    if stock_data is not None:
        stock_data.reset_index(inplace=True)
        df_lstm = stock_data[["Date", "Close"]]

        # Compute a simple moving average (SMA)
        df_lstm["SMA_50"] = df_lstm["Close"].rolling(window=50).mean()

        # Normalize the 'Close' prices
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_lstm["Close_Scaled"] = scaler.fit_transform(df_lstm[["Close"]])

        # Step 2: Allow user to set parameters
        st.sidebar.header("Model Parameters")
        lookback = st.sidebar.slider("Lookback Period (Days)", min_value=10, max_value=100, value=60)
        epochs = st.sidebar.slider("Number of Epochs", min_value=10, max_value=100, value=50)
        batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

        # Prepare the dataset for LSTM
        X, y = [], []
        for i in range(lookback, len(df_lstm)):
            X.append(df_lstm["Close_Scaled"].values[i - lookback:i])
            y.append(df_lstm["Close_Scaled"].values[i])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Step 3: Build and train the LSTM model
        if st.button("Train Model"):
            with st.spinner("Training the model..."):
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25),
                    Dense(1)
                ])

                model.compile(optimizer="adam", loss="mean_squared_error")
                history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

            st.success("Model training completed!")

            # Step 4: Predict the next day's stock price
            test_data = df_lstm["Close_Scaled"].values[-lookback:].reshape(1, lookback, 1)
            predicted_price_scaled = model.predict(test_data)
            predicted_price = scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))

            # Display the predicted price
            st.subheader("Prediction Results")
            st.write(f"Predicted Closing Price: {predicted_price[0][0]:.2f}")

            # Step 5: Visualize actual vs. predicted prices with moving average
            st.subheader("Visualization")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_lstm["Date"], df_lstm["Close"], label="Actual Price")
            ax.plot(df_lstm["Date"], df_lstm["SMA_50"], label="50-Day SMA", linestyle="dashed")
            ax.scatter(df_lstm["Date"].iloc[-1], predicted_price, color='red', label="Predicted Price", marker='o')
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price")
            ax.legend()
            ax.set_title(f"{ticker} Stock Price Prediction using LSTM")
            st.pyplot(fig)

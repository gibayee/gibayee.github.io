# Real-Time Stock Price Prediction using LSTM

## Overview
This project uses a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical data. It fetches real-time stock data from Yahoo Finance, preprocesses it, and trains an LSTM model to predict future stock prices. The model also incorporates a 50-day simple moving average (SMA) as a benchmark.

## Features
- Fetches real-time stock data using Yahoo Finance (`yfinance`)
- Allows users to input any stock ticker symbol
- Normalizes data for better training performance
- Uses LSTM for time-series forecasting
- Provides visualization of actual vs. predicted prices with a moving average benchmark
- User-configurable parameters for lookback period, epochs, and batch size

## Installation
Ensure you have Python installed, then install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage
Run the script to start training and predicting stock prices:

```sh
python stock_price_prediction.py
```

### Parameters
- **Stock Ticker**: Enter any valid stock ticker (e.g., AAPL, TSLA, MSFT).
- **Lookback Period**: Defines how many past days are used for prediction.
- **Epochs**: Number of training iterations.
- **Batch Size**: Size of data batches for model training.

## Model Performance
The model uses Mean Squared Error (MSE) as the loss function and Adam optimizer. It evaluates performance based on training loss and how well predictions align with historical stock prices.

## Notes
- Since the model fetches real-time data, it may take **about 2 minutes to load**, depending on internet speed.
- Predictions are influenced by market trends and may not be 100% accurate.

## Dependencies
The project requires the following Python libraries:

```
numpy
pandas
matplotlib
tensorflow
scikit-learn
yfinance
```

## License
This project is open-source and available under the MIT License.

## Author
[Godfred Bayee](https://linkedin.com/in/godfredbayee/)


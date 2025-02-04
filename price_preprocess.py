# Import external libraries
import numpy as np
import pandas as pd
import math
import time
import datetime

# Import utility functions
import util.analysis as a

'''
Bitcoin Price Analysis
Data Loading and Preprocessing
'''
def preprocess_price_data():
  # Read raw bitcoin dataset
  btc_data = pd.read_csv("data/crytpo_data.csv", index_col = 0)

  # Drop rows with missing values
  btc_data = btc_data.dropna()

  # Convert the 'timestamp' column to datetime and set it as the index
  btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'], unit='s')
  btc_data = btc_data.set_index('timestamp').sort_index()  # Sort by timestamp in ascending order

  # Resample to 6-hour intervals
  daily_data = btc_data.resample('6h').agg({
      'open': 'first',
      'price': 'last',
      'dayHigh': 'max',
      'dayLow': 'min',
      'volume': 'sum'
  })

  # Clean and rename columns for mplfinance
  daily_data = daily_data.fillna(method='ffill').dropna()
  daily_data = daily_data.rename(columns={
      'price': 'Close',
      'dayHigh': 'High',
      'dayLow': 'Low',
      'open': 'Open',
      'volume': 'Volume'
  })

  # Add moving averages
  daily_data['SMA5'] = daily_data['Close'].rolling(window=5, min_periods=1).mean()
  daily_data['SMA10'] = daily_data['Close'].rolling(window=10, min_periods=1).mean()

  # Example usage
  df_with_indicators = a.calculate_technical_indicators(btc_data)

  # If you want pattern detection (only if you have OHLC data):
  if all(col in btc_data.columns for col in ['open', 'close', 'dayHigh', 'dayLow']):
      df_with_indicators = a.detect_patterns(df_with_indicators)

  # Clean hourly BTC price data
  hr_btc = df_with_indicators
  hr_btc['time'] = pd.to_datetime(hr_btc['time'], errors='coerce')
  hr_btc = hr_btc.dropna(subset=['time'])
  hr_btc['time'] = hr_btc['time'].dt.floor('h')

  # Save preprocessed data
  hr_btc.to_csv('./processed/hourly_btc_tw_data.csv')

  return

def convert2_daily_price():
  # Load the hourly dataset
  btc_data = pd.read_csv('./processed/hourly_btc_tw_data.csv')

  # Convert 'time' to datetime
  btc_data['time'] = pd.to_datetime(btc_data['time'])

  # Resample to daily intervals and calculate required metrics
  day_btc = btc_data.resample('D', on='time').agg({
      'price': ['first', 'last', 'mean'],  # Open, Close, and Average Price
      'volume': 'sum',                     # Total Volume
      'dayHigh': 'max',                    # Daily High Price
      'dayLow': 'min',                     # Daily Low Price
      'SMA_5': 'mean',                     # Average SMA_5
      'SMA_10': 'mean',                    # Average SMA_10
      'RSI': 'mean',                       # Average RSI
      'MACD': 'last',                      # Last MACD Value of the Day
      'Signal_Line': 'last',               # Last Signal Line of the Day
      'MACD_Histogram': 'last'             # Last MACD Histogram of the Day
  }).reset_index()

  # Fix for multi-level columns
  day_btc.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in day_btc.columns]

  # Reformat column names
  day_btc = day_btc.rename(columns={
    'time_': 'time',
    'price_first': 'open',
    'price_last': 'close',
    'price_mean': 'price',
    'volume_sum': 'volume',
    'dayHigh_max': 'dayHigh',
    'dayLow_min': 'dayLow',
    'SMA_5_mean': 'SMA_5',
    'SMA_10_mean': 'SMA_10',
    'RSI_mean': 'RSI',
    'MACD_last': 'MACD',
    'Signal_Line_last': 'Signal_Line',
    'MACD_Histogram_last': 'MACD_Histogram'
  })

  # Save the daily data to a CSV file
  day_btc.to_csv('./processed/day_btc_data.csv', index=False)

  return day_btc

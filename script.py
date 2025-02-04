import sys
# Import external libraries
import numpy as np
import pandas as pd
import math
import time
import datetime
# import mplfinance as mpf
from dateutil import parser
import re

# Import utility functions
import util.analysis as a
import util.vader as t
from util.train import split_data, scale_features
from util.random_forest import train_model, evaluate_model

if __name__ == '__main__': 
  '''
  Bitcoin Price Analysis
  Data Loading and Preprocessing
  '''
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

  '''
  Twitter Sentimental Analysis
  '''
  # Read raw twitter dataset
  tw_data = pd.read_csv('data/twitter_data.csv',index_col=0)

  # If there is missing values, drop these missing values
  tw_data = tw_data.dropna()

  # Extract link values from the **text** column with regex.
  tw_data['text'] = tw_data['text'].apply(
    lambda x: re.sub(r'https?://\S+', '', x).strip()
    )  

  # Remove all "\n" from the **text** column.
  tw_data['text'] = tw_data['text'].replace('\n', '', regex=True)

  # Assert if there is any empty strings for the **text** column
  tw_data = tw_data[tw_data['text'] != '']

  # Drop every row where **text** column is an empty string
  tw_data = tw_data.drop(tw_data[tw_data['text'].isna() | (tw_data['text'].str.strip() == '')].index)

  # Drop rows where column quotes or replies or retweets or bookmarks or favorites is less than 10.
  tw_data = tw_data.drop(
    tw_data[
      (tw_data['quotes'] < 10) | 
      (tw_data['replies'] < 10) | 
      (tw_data['retweets'] < 10) | 
      (tw_data['bookmarks'] < 10) | 
      (tw_data['favorites'] < 10)
    ].index
  )

  # Drop rows where column lang is not "en" (Twitter text is not in English)
  tw_data = tw_data.drop(tw_data[(tw_data['lang'] != 'en')].index)

  # Apply VADER sentiment anaylysis to the twitter dataset.
  tw_data[['vd_negative', 'vd_neutral', 'vd_positive', 'vd_compound']] = tw_data['text'].apply(
    lambda x: pd.Series(t.vader_sentiment(x))
    )

  # Convert current cleaned data to csv
  tw_data.to_csv('./processed/processed_twitter_data.csv', index=False)

  '''
  Continue Preprocess VADER Twitter Sentiment Data
  '''
  # Read processed Twitter dataset
  tw_data = pd.read_csv('./processed/processed_twitter_data.csv')

  # Convert **time** column datatype
  tw_data['time'] = pd.to_datetime(tw_data['time'], format='mixed')
  tw_data = tw_data.sort_values(by='time')

  # Hourly mean
  v_hr_mean = tw_data.groupby(
    tw_data['time'].dt.floor('h')
    )[['vd_positive', 'vd_negative', 'vd_neutral', 'vd_compound']].mean().reset_index()

  # Hourly median
  v_hr_med = tw_data.groupby(
    tw_data['time'].dt.floor('h')
    )[['vd_positive', 'vd_negative', 'vd_neutral', 'vd_compound']].median().reset_index()

  # Daily mean
  v_day_mean = tw_data.groupby(
    tw_data['time'].dt.floor('d')
    )[['vd_positive', 'vd_negative', 'vd_neutral', 'vd_compound']].mean().reset_index()
  
  # Daily median
  v_day_med = tw_data.groupby(
    tw_data['time'].dt.floor('d')
    )[['vd_positive', 'vd_negative', 'vd_neutral', 'vd_compound']].median().reset_index()

  '''
  Find Correlation between Bitcoin Price Movemnent VS. Twitter Sentiment
  '''

  # Load the hourly dataset
  btc_data = pd.read_csv('./processed/hourly_btc_tw_data.csv')

  # Convert 'time' to datetime
  btc_data['time'] = pd.to_datetime(btc_data['time'])
  btc_data.head(3)

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

  # # Display the first few rows of the resampled daily data
  # print("\nFirst few rows:")
  # print(day_btc.head())

  # # Check the data types and missing values
  # print("\nDataset Info:")
  # print(day_btc.info())

  # print("\nMissing Values:")
  # print(day_btc.isnull().sum())

  # Save the daily data to a CSV file
  day_btc.to_csv('./processed/day_btc_data.csv', index=False)

  # Merge daily btc_data with tw_data using sentiment mean values
  day_btc_tw = pd.merge(day_btc, v_day_mean, on='time')

  day_btc_tw = day_btc_tw.rename(columns={
    "vd_positive": "vd_positive_mean",
    'vd_neutral': "vd_neutral_mean",
    "vd_negative": "vd_negative_mean",
    "vd_compound": "vd_compound_mean",
  })

  # Merge btc_data with tw_data using sentiment median values
  day_btc_tw = pd.merge(day_btc_tw, v_day_med, on='time')

  # Rename columns
  day_btc_tw = day_btc_tw.rename(columns={
    "vd_positive": "vd_positive_med",
    'vd_neutral': "vd_neutral_med",
    "vd_negative": "vd_negative_med",
    "vd_compound": "vd_compound_med",
  })

  # Save the daily data to a CSV file if needed
  day_btc_tw = day_btc_tw.dropna()
  day_btc_tw.to_csv('./processed/day_btc_tw.csv', index=False)

  '''
  Random Forest Model
  '''
  # Load and preprocess data
  df = pd.read_csv('./processed/day_btc_tw.csv')
  df['time'] = pd.to_datetime(df['time'])
  df.set_index('time', inplace=True)

  # open,close,price,volume,dayHigh,dayLow,SMA_5,SMA_10,RSI,MACD,Signal_Line,MACD_Histogram,vd_positive_mean,vd_negative_mean,vd_neutral_mean,vd_compound_mean,vd_positive_med,vd_negative_med,vd_neutral_med,vd_compound_med
  # Feature sets
  technical_features = ['SMA_5', 'SMA_10', 'RSI', 'MACD']
  sentiment_features = ['vd_neutral_mean', 
        'vd_negative_med', 'vd_neutral_med',
        'vd_negative_mean',
        'vd_positive_med', 'vd_compound_med', 'vd_compound_mean', 
        'vd_positive_mean']
  price_features = ['price','volume', 'dayHigh', 'dayLow']

  df['volatility'] = df['price'].pct_change().rolling(window=10).std() * 100  # Rolling std dev of price changes

  X = df[technical_features + sentiment_features + price_features + ['volatility']].copy()
  y = df['price'].shift(-1)  # Predict next day's price 

  # Create a binary target variable: 1 if price increases, 0 if price decreases
  df['price_change'] = df['price'].shift(-1) - df['price']
  df['target'] = (df['price_change'] > 0).astype(int)  # 1 for increase, 0 for decrease

  # Remove NaN values from both X and y
  df_clean = pd.concat([X, df['target'].rename('target')], axis=1).dropna()

  # Ensure 'price' is a single Series
  if isinstance(df_clean['price'], pd.DataFrame):
      df_clean['price'] = df_clean['price'].iloc[:, 0]  # Select the first column

  X_train, X_test, y_train, y_test = split_data(df_clean, technical_features, sentiment_features, price_features)
  X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

  model = train_model(X_train_scaled, y_train)
  accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test_scaled, y_test)










  

    
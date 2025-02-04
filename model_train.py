# Import external libraries
import numpy as np
import pandas as pd
import math
import time
import datetime
# Import utility functions
from util.train import split_data, scale_features
from util.random_forest import train_model, evaluate_model

def train_random_forest_model():
  # Load and preprocess data
  df = pd.read_csv('./processed/day_btc_tw.csv')
  df['time'] = pd.to_datetime(df['time'])
  df.set_index('time', inplace=True)
  
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

